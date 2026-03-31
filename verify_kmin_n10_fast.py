"""
Fast vectorized k=6 classification test for n=10.
Processes ALL C(n,k) subsets at once using tensor ops — no per-subset loop.
"""
import torch
import numpy as np
from itertools import combinations, permutations
from collections import defaultdict
import time
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'graph_data')


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
    print(f"  Building k={k} lookup ({1<<ne} patterns, {k}!={720 if k==6 else 120} perms)...", end='', flush=True)
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
    return arr, type_id


def compute_sigs_fast(graphs_np, n, k, lookup_np, num_types, batch_size=8192):
    """
    Fully vectorized: process all C(n,k) subsets at once per batch.

    For each batch of graphs (B, n, n):
    1. Compute edge patterns for ALL subsets simultaneously: (B, num_subsets)
    2. Look up type IDs: (B, num_subsets)
    3. Bincount to get (B, num_types)
    """
    N = len(graphs_np)
    subsets = list(combinations(range(n), k))
    num_subs = len(subsets)
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    ne = len(edges)

    # Precompute subset vertex indices: (num_subs, k)
    subs_arr = torch.tensor(subsets, device=DEVICE, dtype=torch.long)  # (S, k)

    # Precompute edge index pairs in local coords: (ne, 2)
    edge_li = torch.tensor([li for (li, lj) in edges], device=DEVICE, dtype=torch.long)
    edge_lj = torch.tensor([lj for (li, lj) in edges], device=DEVICE, dtype=torch.long)

    # Global indices: gi[e, s] = subs_arr[s, edge_li[e]]
    gi = subs_arr[:, edge_li].t()  # (ne, S)
    gj = subs_arr[:, edge_lj].t()  # (ne, S)

    # Powers of 2 for bit encoding: (ne, 1)
    powers = torch.tensor([1 << e for e in range(ne)], device=DEVICE, dtype=torch.long).unsqueeze(1)  # (ne, 1)

    lookup_t = torch.tensor(lookup_np, device=DEVICE, dtype=torch.long)
    G_all = torch.tensor(graphs_np, device=DEVICE, dtype=torch.long)  # (N, n, n)

    sigs = np.zeros((N, num_types), dtype=np.int32)
    t0 = time.time()

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        B = batch_end - batch_start
        G_b = G_all[batch_start:batch_end]  # (B, n, n)

        # Extract edges for all subsets at once
        # G_b[:, gi, gj] -> (B, ne, S)
        edge_vals = G_b[:, gi, gj]  # (B, ne, S)

        # Compute bit patterns: (B, S)
        # patterns[b, s] = sum_e edge_vals[b, e, s] * 2^e
        patterns = (edge_vals * powers.unsqueeze(0)).sum(dim=1)  # (B, S)

        # Look up type IDs: (B, S)
        type_ids = lookup_t[patterns]  # (B, S)

        # Bincount: for each graph, count each type
        # Using scatter_add: (B, num_types)
        type_counts = torch.zeros(B, num_types, device=DEVICE, dtype=torch.int32)
        ones = torch.ones(B * num_subs, device=DEVICE, dtype=torch.int32)
        flat_type_ids = type_ids.reshape(-1)  # (B*S,)
        batch_offsets = torch.arange(B, device=DEVICE).repeat_interleave(num_subs) * num_types
        flat_indices = flat_type_ids + batch_offsets
        type_counts_flat = type_counts.reshape(-1)
        type_counts_flat.scatter_add_(0, flat_indices, ones)

        sigs[batch_start:batch_end] = type_counts.cpu().numpy()

        if (batch_start // batch_size) % 100 == 0:
            elapsed = time.time() - t0
            rate = (batch_end) / elapsed if elapsed > 0 else 0
            eta = (N - batch_end) / rate if rate > 0 else 0
            print(f"    {batch_end:>10,}/{N:,}  {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

    print(f"    Done in {time.time()-t0:.1f}s")
    return sigs


def check_collisions(sigs, label):
    sig_list = [tuple(row) for row in sigs]
    groups = defaultdict(list)
    for i, s in enumerate(sig_list):
        groups[s].append(i)
    hard = [(s, idxs) for s, idxs in groups.items() if len(idxs) > 1]
    n_distinct = len(groups)
    N = len(sigs)
    print(f"  {label}: {n_distinct:,}/{N:,} distinct, {len(hard)} collision groups")
    if hard:
        for _, idxs in hard[:3]:
            print(f"    pair {idxs[0]},{idxs[1]}")
    return len(hard)


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # n=10
    g10_file = os.path.join(DATA_DIR, 'graph10_decompressed.g6')
    print(f"\nLoading n=10 graphs...")
    t0 = time.time()
    g10 = load_g6(g10_file)
    print(f"  Loaded {len(g10):,} graphs in {time.time()-t0:.1f}s")
    EXPECTED = 12005168
    if len(g10) != EXPECTED:
        print(f"  WARNING: expected {EXPECTED:,}")

    graphs_np = np.stack(g10, axis=0)  # (N, 10, 10)
    N = len(graphs_np)

    # Build k=6 lookup
    lookup6, num6 = build_type_lookup(6)

    # Compute k=6 sigs
    print(f"\nComputing k=6 induced subgraph sigs for {N:,} graphs (n=10)...")
    print(f"  Subsets: C(10,6)={len(list(combinations(range(10),6)))}, Types: {num6}")
    sigs6 = compute_sigs_fast(graphs_np, 10, 6, lookup6, num6, batch_size=4096)

    n_coll = check_collisions(sigs6, "k<=6")
    if n_coll == 0:
        print(f"\n*** COMPLETE: k_min(10) <= 6 ***")
        print(f"*** k=6 is sufficient for n=10! ***")
    else:
        print(f"\n{n_coll} collision groups remain. Testing k=5+6 combined...")
        # Also compute k=5
        lookup5, num5 = build_type_lookup(5)
        sigs5 = compute_sigs_fast(graphs_np, 10, 5, lookup5, num5, batch_size=8192)
        sigs56 = np.concatenate([sigs5, sigs6], axis=1)
        n_coll2 = check_collisions(sigs56, "k<=5+6")
        print(f"\nResult: {'COMPLETE' if n_coll2==0 else f'{n_coll2} groups remain'}")
