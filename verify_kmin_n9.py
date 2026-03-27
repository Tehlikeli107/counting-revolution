"""
Verify k_min(n) for n=9 and n=10 using correct canonical form (GPU).

New finding: k_min(7)=n-2=5, k_min(8)=n-2=6.
Question: is k_min(9)=n-3=6 or n-2=7?
         is k_min(10)=n-3=7 or n-2=8?
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
    A = np.zeros((n, n), dtype=np.int8)
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
    """Map EVERY edge bit pattern -> canonical type id."""
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    ne = len(edges)
    print(f"    Building k={k} lookup ({1<<ne} patterns, {k}!={sum(1 for _ in permutations(range(k)))} perms)...", end='', flush=True)
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


def compute_sigs_gpu(graphs_np, n, k, lookup_np, num_types, batch_size=1024):
    """GPU batch induced k-subgraph counting."""
    N = len(graphs_np)
    subsets = list(combinations(range(n), k))
    num_subs = len(subsets)
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]

    lookup_t = torch.tensor(lookup_np, device=DEVICE, dtype=torch.int64)
    sigs = np.zeros((N, num_types), dtype=np.int32)

    G_all = torch.tensor(graphs_np, device=DEVICE, dtype=torch.int64)  # (N, n, n)

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        B = batch_end - batch_start
        G_batch = G_all[batch_start:batch_end]  # (B, n, n)
        type_counts = torch.zeros(B, num_types, device=DEVICE, dtype=torch.int32)

        for si in range(num_subs):
            sub = subsets[si]
            pattern = torch.zeros(B, device=DEVICE, dtype=torch.int64)
            for ei, (li, lj) in enumerate(edges):
                gi, gj = sub[li], sub[lj]
                pattern = pattern | (G_batch[:, gi, gj] << ei)
            type_ids = lookup_t[pattern]
            type_counts.scatter_add_(
                1,
                type_ids.unsqueeze(1),
                torch.ones(B, 1, device=DEVICE, dtype=torch.int32)
            )
        sigs[batch_start:batch_end] = type_counts.cpu().numpy()

    return sigs


def analyze(n, graphs, k_list):
    graphs_np = np.stack(graphs, axis=0)
    N = len(graphs)
    print(f"\n{'='*60}")
    print(f"  n={n}: {N:,} graphs  Device={DEVICE}")
    print(f"{'='*60}")

    cumulative_sigs = np.zeros((N, 0), dtype=np.int32)

    for k in k_list:
        lookup, num_types = build_type_lookup(k)

        t0 = time.time()
        sigs_k = compute_sigs_gpu(graphs_np, n, k, lookup, num_types)
        elapsed = time.time() - t0

        cumulative_sigs = np.concatenate([cumulative_sigs, sigs_k], axis=1)

        # Count collisions
        sig_list = [tuple(row) for row in cumulative_sigs]
        groups = defaultdict(list)
        for i, s in enumerate(sig_list):
            groups[s].append(i)
        n_distinct = len(groups)
        hard = [(s, idxs) for s, idxs in groups.items() if len(idxs) > 1]
        n_coll_groups = len(hard)

        print(f"\n  k<={k}: {n_distinct:,}/{N:,} distinct, "
              f"{n_coll_groups} collision groups  [{elapsed:.1f}s]")

        if n_coll_groups == 0:
            print(f"  *** COMPLETE: k_min({n}) <= {k} ***")
            break
        else:
            for _, idxs in hard[:3]:
                g1 = graphs_np[idxs[0]]
                e1 = int(g1.sum()) // 2
                d1 = tuple(sorted(g1.sum(axis=1).astype(int).tolist()))
                print(f"    pair {idxs[0]},{idxs[1]}: {e1} edges  deg={d1}")
            if len(hard) > 3:
                print(f"    ... and {len(hard)-3} more groups")
    else:
        print(f"  k_min({n}) > {k_list[-1]}  (tested up to k={k_list[-1]})")


if __name__ == '__main__':
    # n=9
    g9_file = os.path.join(DATA_DIR, 'graph9.g6')
    if os.path.exists(g9_file):
        print(f"Loading n=9 graphs...")
        t0 = time.time()
        g9 = load_g6(g9_file)
        print(f"  Loaded {len(g9):,} graphs in {time.time()-t0:.1f}s")
        analyze(9, g9, k_list=[5, 6, 7])
    else:
        print(f"graph9.g6 not found at {g9_file}")

    # n=10
    g10_file = os.path.join(DATA_DIR, 'graph10_decompressed.g6')
    if os.path.exists(g10_file):
        print(f"\nLoading n=10 graphs...")
        t0 = time.time()
        g10 = load_g6(g10_file)
        print(f"  Loaded {len(g10):,} graphs in {time.time()-t0:.1f}s")
        analyze(10, g10, k_list=[6, 7, 8])
    else:
        print(f"graph10.g6 not found at {g10_file}")
