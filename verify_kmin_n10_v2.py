"""
Memory-efficient k=6 test for n=10.
- Batch GPU processing (don't load all 12M onto GPU at once)
- Numpy sort-based collision detection (no Python tuple overhead)
"""
import torch
import numpy as np
from itertools import combinations, permutations
import time
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'graph_data')
LOOKUP_CACHE = os.path.join(DATA_DIR, 'k6_lookup.npy')


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


def build_type_lookup(k):
    """Correct: maps ALL bit patterns to canonical type."""
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    ne = len(edges)
    if os.path.exists(LOOKUP_CACHE) and k == 6:
        arr = np.load(LOOKUP_CACHE)
        type_id = int(arr.max()) + 1
        print(f"  Loaded k={k} lookup from cache: {type_id} types")
        return arr, type_id

    print(f"  Building k={k} lookup ({1<<ne} patterns)...", end='', flush=True)
    t0 = time.time()
    canon_to_type = {}
    type_id = 0
    arr = np.zeros(1 << ne, dtype=np.int32)
    for bits in range(1 << ne):
        A = np.zeros((k, k), dtype=np.int8)
        for idx, (i, j) in enumerate(edges):
            if (bits >> idx) & 1:
                A[i, j] = A[j, i] = 1
        min_bits = min(
            sum(int(A[perm[i], perm[j]]) << eidx
                for eidx, (i, j) in enumerate(edges))
            for perm in permutations(range(k))
        )
        if min_bits not in canon_to_type:
            canon_to_type[min_bits] = type_id
            type_id += 1
        arr[bits] = canon_to_type[min_bits]
    print(f" {type_id} types, {time.time()-t0:.1f}s")
    if k == 6:
        np.save(LOOKUP_CACHE, arr)
        print(f"  Saved k=6 lookup to cache")
    return arr, type_id


def compute_sigs_batched(graphs_np, n, k, lookup_np, num_types, batch_size=2048):
    """
    Process batches of graphs. For each batch:
    - Load only batch_size graphs onto GPU
    - Vectorize over all C(n,k) subsets simultaneously
    """
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

    sigs = np.zeros((N, num_types), dtype=np.int32)
    t0 = time.time()

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        B = batch_end - batch_start

        G_b = torch.tensor(graphs_np[batch_start:batch_end],
                           device=DEVICE, dtype=torch.long)  # (B, n, n)

        # Edge values for all subsets: (B, ne, S)
        edge_vals = G_b[:, gi, gj]

        # Bit patterns: (B, S)
        patterns = (edge_vals * powers.unsqueeze(0)).sum(dim=1)

        # Type IDs: (B, S)
        type_ids = lookup_t[patterns]

        # Bincount per graph: (B, num_types)
        # Use scatter_add on flattened tensors
        type_counts = torch.zeros(B * num_types, device=DEVICE, dtype=torch.int32)
        offsets = torch.arange(B, device=DEVICE, dtype=torch.long).repeat_interleave(num_subs) * num_types
        flat_ids = type_ids.reshape(-1) + offsets
        ones = torch.ones(B * num_subs, device=DEVICE, dtype=torch.int32)
        type_counts.scatter_add_(0, flat_ids, ones)

        sigs[batch_start:batch_end] = type_counts.view(B, num_types).cpu().numpy()

        if batch_start % (batch_size * 200) == 0:
            elapsed = time.time() - t0
            rate = batch_end / elapsed if elapsed > 0.1 else 1
            eta = (N - batch_end) / rate
            print(f"  {batch_end:>10,}/{N:,}  {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

    print(f"  Done: {time.time()-t0:.1f}s total")
    return sigs


def count_collisions_hash(sigs):
    """Hash-based O(N*D) collision detection — no huge sort needed."""
    N, D = sigs.shape
    print(f"  Hashing {N:,} x {D} signature matrix...", end='', flush=True)
    t0 = time.time()
    # Polynomial hash mod 2^64 using vectorized numpy
    h = np.zeros(N, dtype=np.uint64)
    for j in range(D):
        h = h * np.uint64(1000003) + sigs[:, j].astype(np.uint64)
    print(f" {time.time()-t0:.2f}s")

    # Find unique hashes
    print(f"  Sorting hashes...", end='', flush=True)
    t0 = time.time()
    order = np.argsort(h)
    sorted_h = h[order]
    print(f" {time.time()-t0:.2f}s")

    # Find hash duplicates (possible collisions)
    hash_dup = sorted_h[1:] == sorted_h[:-1]
    n_hash_coll = int(hash_dup.sum())
    n_distinct_hash = N - n_hash_coll

    print(f"  {n_distinct_hash:,}/{N:,} distinct hashes, {n_hash_coll} hash collisions")

    if n_hash_coll == 0:
        print(f"  ZERO COLLISIONS (hash-verified)")
        return 0

    # For hash collisions, verify exact equality
    print(f"  Verifying {n_hash_coll} hash collisions exactly...")
    coll_groups = 0
    i = 0
    while i < N - 1:
        if sorted_h[i] == sorted_h[i+1]:
            # Find extent of this hash group
            j = i + 1
            while j < N and sorted_h[j] == sorted_h[i]:
                j += 1
            # Check if all rows in group are actually equal
            idxs = order[i:j]
            rows = sigs[idxs]
            # All same? Check pairwise
            is_dup = np.all(rows == rows[0], axis=1)
            if is_dup.all():
                coll_groups += 1
                print(f"    TRUE collision group: {idxs.tolist()}")
            else:
                # Hash collision but different sigs (rare)
                pass
            i = j
        else:
            i += 1

    print(f"  {N - coll_groups:,}/{N:,} distinct, {coll_groups} true collision groups")
    return coll_groups


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    g10_file = os.path.join(DATA_DIR, 'graph10_decompressed.g6')
    print(f"\nLoading n=10 graphs...")
    t0 = time.time()
    g10 = load_g6(g10_file)
    N = len(g10)
    print(f"  Loaded {N:,} graphs in {time.time()-t0:.1f}s")

    graphs_np = np.stack(g10, axis=0)  # (N, 10, 10) uint8
    del g10  # free memory

    # Build k=6 lookup
    lookup6, num6 = build_type_lookup(6)

    print(f"\nComputing k=6 sigs for n=10 ({N:,} graphs, C(10,6)=210 subsets, {num6} types)...")
    sigs6 = compute_sigs_batched(graphs_np, 10, 6, lookup6, num6, batch_size=2048)

    print(f"\nChecking collisions (k<=6):")
    n_coll = count_collisions_hash(sigs6)

    if n_coll == 0:
        print(f"\n*** PROVEN: k_min(10) <= 6 ***")
        print(f"*** k=6 induced subgraph counting classifies ALL n=10 graphs! ***")
        print(f"*** CONJECTURE: k=6 is universal for all n>=8 ***")
    else:
        print(f"\n{n_coll} collision groups. k=6 is NOT sufficient for n=10.")
        print(f"Need to test k=7...")
