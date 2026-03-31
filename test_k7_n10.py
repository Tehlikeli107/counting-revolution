"""
Test k=7 for n=10 (22 collision groups survive k=6).
Uses vectorized numpy to build k=7 lookup (2M entries, 1044 types).
"""
import torch
import numpy as np
from itertools import combinations, permutations
import time
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'graph_data')
LOOKUP7_CACHE = os.path.join(DATA_DIR, 'k7_lookup.npy')
SIGS6_CACHE = os.path.join(DATA_DIR, 'n10_sigs6.npy')


def parse_graph6(line):
    line = line.strip()
    if line.startswith('>>graph6<<'):
        line = line[10:]
    data = [ord(c) - 63 for c in line]
    if data[0] <= 62:
        n = data[0]; bits_start = 1
    else:
        n = ((data[1]&63)<<12)|((data[2]&63)<<6)|(data[3]&63); bits_start = 4
    A = np.zeros((n, n), dtype=np.uint8)
    bit_idx = 0
    for j in range(1, n):
        for i in range(j):
            bp = bits_start + bit_idx // 6
            bw = 5 - (bit_idx % 6)
            if bp < len(data) and (data[bp] >> bw) & 1:
                A[i, j] = A[j, i] = 1
            bit_idx += 1
    return A


def load_g6(filepath, max_graphs=None):
    graphs = []
    with open(filepath, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            if max_graphs and i >= max_graphs:
                break
            if line.strip():
                graphs.append(parse_graph6(line))
    return graphs


def build_k7_lookup_fast():
    """Vectorized k=7 lookup: 2^21 patterns -> 1044 type IDs."""
    if os.path.exists(LOOKUP7_CACHE):
        arr = np.load(LOOKUP7_CACHE)
        n_types = int(arr.max()) + 1
        print(f"  Loaded k=7 lookup from cache: {n_types} types")
        return arr, n_types

    k = 7
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    ne = len(edges)  # 21
    N_pats = 1 << ne  # 2M

    print(f"  Building k=7 lookup ({N_pats:,} patterns, {k}!={5040} perms)...", flush=True)
    t0 = time.time()

    # Pre-extract all ne bits for all 2M patterns: (N_pats, ne) int8
    all_pats = np.arange(N_pats, dtype=np.int64)
    all_bits = ((all_pats[:, None] >> np.arange(ne, dtype=np.int64)[None, :]) & 1).astype(np.int8)
    # all_bits[p, e] = 1 if bit e is set in pattern p

    edge_idx = {(i,j): e for e, (i,j) in enumerate(edges)}
    powers = (1 << np.arange(ne, dtype=np.int64))

    min_pats = all_pats.copy()

    all_perms = list(permutations(range(k)))
    print(f"  Precomputed base data in {time.time()-t0:.1f}s, iterating {len(all_perms)} perms...", flush=True)

    t1 = time.time()
    for pi, perm in enumerate(all_perms):
        # For this perm, compute src_to_dst: original bit e_src -> new bit e_dst
        # Edge (i,j) under perm maps to (perm[i], perm[j]) normalized
        src_to_dst = np.zeros(ne, dtype=np.int64)
        for e_src, (i, j) in enumerate(edges):
            pi2, pj2 = perm[i], perm[j]
            if pi2 > pj2: pi2, pj2 = pj2, pi2
            e_dst = edge_idx[(pi2, pj2)]
            src_to_dst[e_src] = e_dst

        # Rearrange bits: bits in new positions
        # new_bits[p, e_dst] = all_bits[p, e_src] where e_dst = src_to_dst[e_src]
        # perm_pats[p] = sum_e all_bits[p, e_src] * powers[src_to_dst[e_src]]
        perm_pats = (all_bits * powers[src_to_dst][None, :]).sum(axis=1)

        min_pats = np.minimum(min_pats, perm_pats)

        if (pi + 1) % 504 == 0:
            print(f"    {pi+1}/{len(all_perms)} perms  {time.time()-t1:.0f}s", flush=True)

    # Assign type IDs
    unique_canon, inverse = np.unique(min_pats, return_inverse=True)
    n_types = len(unique_canon)
    arr = inverse.astype(np.int32)

    elapsed = time.time() - t0
    print(f"  Done: {n_types} types in {elapsed:.1f}s")
    np.save(LOOKUP7_CACHE, arr)
    print(f"  Saved to cache")
    return arr, n_types


def compute_sigs_batched(graphs_np, n, k, lookup_np, num_types, batch_size=2048):
    """Generic batched GPU sig computation (for k=6)."""
    N = len(graphs_np)
    subsets = list(combinations(range(n), k))
    num_subs = len(subsets)
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    ne = len(edges)
    subs_arr = torch.tensor(subsets, device=DEVICE, dtype=torch.long)
    edge_li = torch.tensor([li for (li, lj) in edges], device=DEVICE, dtype=torch.long)
    edge_lj = torch.tensor([lj for (li, lj) in edges], device=DEVICE, dtype=torch.long)
    gi = subs_arr[:, edge_li].t()
    gj = subs_arr[:, edge_lj].t()
    powers = torch.tensor([1 << e for e in range(ne)], device=DEVICE, dtype=torch.long).view(ne, 1)
    lookup_t = torch.tensor(lookup_np, device=DEVICE, dtype=torch.long)
    sigs = np.zeros((N, num_types), dtype=np.int32)
    t0 = time.time()
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        B = batch_end - batch_start
        G_b = torch.tensor(graphs_np[batch_start:batch_end], device=DEVICE, dtype=torch.long)
        edge_vals = G_b[:, gi, gj]
        patterns = (edge_vals * powers.unsqueeze(0)).sum(dim=1)
        type_ids = lookup_t[patterns]
        type_counts = torch.zeros(B * num_types, device=DEVICE, dtype=torch.int32)
        offsets = torch.arange(B, device=DEVICE, dtype=torch.long).repeat_interleave(num_subs) * num_types
        ones = torch.ones(B * num_subs, device=DEVICE, dtype=torch.int32)
        type_counts.scatter_add_(0, type_ids.reshape(-1) + offsets, ones)
        sigs[batch_start:batch_end] = type_counts.view(B, num_types).cpu().numpy()
        if batch_start % (batch_size * 200) == 0:
            elapsed = time.time() - t0
            rate = batch_end / elapsed if elapsed > 0.1 else 1
            print(f"  k={k}: {batch_end:>10,}/{N:,}  ETA {(N-batch_end)/rate:.0f}s", flush=True)
    print(f"  k={k} done: {time.time()-t0:.1f}s")
    return sigs


def compute_sigs_k7(graphs_np, lookup_np, num_types, batch_size=1024):
    """Compute k=7 induced subgraph sigs for all n=10 graphs."""
    n = 10; k = 7
    N = len(graphs_np)
    subsets = list(combinations(range(n), k))
    num_subs = len(subsets)
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    ne = len(edges)

    subs_arr = torch.tensor(subsets, device=DEVICE, dtype=torch.long)  # (S, k)
    edge_li = torch.tensor([li for (li, lj) in edges], device=DEVICE, dtype=torch.long)
    edge_lj = torch.tensor([lj for (li, lj) in edges], device=DEVICE, dtype=torch.long)
    gi = subs_arr[:, edge_li].t()  # (ne, S)
    gj = subs_arr[:, edge_lj].t()  # (ne, S)
    powers = torch.tensor([1 << e for e in range(ne)], device=DEVICE, dtype=torch.long).view(ne, 1)
    lookup_t = torch.tensor(lookup_np, device=DEVICE, dtype=torch.long)

    print(f"  C({n},{k})={num_subs} subsets, {num_types} types, batch={batch_size}", flush=True)
    sigs = np.zeros((N, num_types), dtype=np.int32)
    t0 = time.time()

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        B = batch_end - batch_start
        G_b = torch.tensor(graphs_np[batch_start:batch_end],
                           device=DEVICE, dtype=torch.long)  # (B, n, n)
        edge_vals = G_b[:, gi, gj]  # (B, ne, S)
        patterns = (edge_vals * powers.unsqueeze(0)).sum(dim=1)  # (B, S) values in 0..2^21
        type_ids = lookup_t[patterns]  # (B, S)
        type_counts = torch.zeros(B * num_types, device=DEVICE, dtype=torch.int32)
        offsets = torch.arange(B, device=DEVICE, dtype=torch.long).repeat_interleave(num_subs) * num_types
        ones = torch.ones(B * num_subs, device=DEVICE, dtype=torch.int32)
        type_counts.scatter_add_(0, type_ids.reshape(-1) + offsets, ones)
        sigs[batch_start:batch_end] = type_counts.view(B, num_types).cpu().numpy()

        if batch_start % (batch_size * 500) == 0:
            elapsed = time.time() - t0
            rate = batch_end / elapsed if elapsed > 0.1 else 1
            eta = (N - batch_end) / rate
            print(f"  {batch_end:>10,}/{N:,}  {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

    print(f"  Done k=7 sigs: {time.time()-t0:.1f}s")
    return sigs


def count_collisions_hash(sigs, label=""):
    N, D = sigs.shape
    print(f"  Hashing {N:,} x {D} sigs ({label})...", end='', flush=True)
    t0 = time.time()
    h = np.zeros(N, dtype=np.uint64)
    for j in range(D):
        h = h * np.uint64(1000003) + sigs[:, j].astype(np.uint64)
    print(f" {time.time()-t0:.2f}s", flush=True)
    order = np.argsort(h)
    sorted_h = h[order]
    hash_dup = sorted_h[1:] == sorted_h[:-1]
    n_hash_coll = int(hash_dup.sum())

    if n_hash_coll == 0:
        print(f"  ZERO collisions (hash verified) -- COMPLETE!")
        return 0

    # Verify
    true_colls = 0; i = 0
    collision_indices = []
    while i < N - 1:
        if sorted_h[i] == sorted_h[i+1]:
            j = i+1
            while j < N and sorted_h[j] == sorted_h[i]: j += 1
            idxs = order[i:j]
            rows = sigs[idxs]
            if np.all(rows == rows[0], axis=1).all():
                true_colls += 1
                collision_indices.append(idxs.tolist())
            i = j
        else: i += 1

    print(f"  {true_colls} true collision groups")
    for idxs in collision_indices[:5]:
        print(f"    graphs {idxs}")
    return true_colls


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load graphs
    print(f"\nLoading n=10 graphs...")
    t0 = time.time()
    g10 = load_g6(os.path.join(DATA_DIR, 'graph10_decompressed.g6'))
    graphs_np = np.stack(g10, axis=0)
    del g10
    N = len(graphs_np)
    print(f"  Loaded {N:,} graphs in {time.time()-t0:.1f}s")

    # Load or compute k=6 sigs
    if os.path.exists(SIGS6_CACHE):
        print(f"\nLoading k=6 sigs from cache...")
        sigs6 = np.load(SIGS6_CACHE)
        print(f"  Loaded {sigs6.shape}")
    else:
        print(f"\nComputing k=6 sigs (needed for combined test)...")
        lookup6 = np.load(os.path.join(DATA_DIR, 'k6_lookup.npy'))
        num6 = int(lookup6.max()) + 1
        print(f"  k=6 lookup: {num6} types")
        sigs6 = compute_sigs_batched(graphs_np, 10, 6, lookup6, num6)
        np.save(SIGS6_CACHE, sigs6)
        print(f"  Saved k=6 sigs to cache")

    # Build k=7 lookup
    print(f"\nBuilding k=7 lookup table...")
    lookup7, num7 = build_k7_lookup_fast()

    # Compute k=7 sigs
    print(f"\nComputing k=7 sigs for {N:,} graphs...")
    sigs7 = compute_sigs_k7(graphs_np, lookup7, num7, batch_size=512)

    # Check collisions on k=7 alone
    n_coll7 = count_collisions_hash(sigs7, "k=7 alone")

    # Check k=6+7 combined (if sigs6 available)
    if sigs6 is not None:
        print(f"\nChecking k=6+7 combined...")
        sigs67 = np.concatenate([sigs6, sigs7], axis=1)
        n_coll67 = count_collisions_hash(sigs67, "k<=7 combined")
        if n_coll67 == 0:
            print(f"\n*** PROVEN: k_min(10) <= 7 ***")
            print(f"*** k<=7 (= n-3) induced subgraph counting classifies ALL n=10 graphs! ***")
        else:
            print(f"\n{n_coll67} groups survive k<=7. k_min(10) > 7.")
    else:
        print(f"\nNote: sigs6 not cached, can't check combined k<=7. Just checking k=7 alone:")
        print(f"k=7 alone: {n_coll7} collision groups")
