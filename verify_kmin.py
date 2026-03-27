"""
Verify k_min(n) for pure induced subgraph counting classification.
Tests n=7 (networkx atlas) and n=8 (McKay graph8.g6).

For n=7: does k=4=n-3 work?
For n=8: does k=5=n-3 work, or do we need k=6=n-2?
"""
import numpy as np
from itertools import combinations, permutations
from collections import defaultdict
import time
import os
import networkx as nx

GRAPH8_FILE = os.path.join(os.path.dirname(__file__), 'graph_data', 'graph8.g6')


# ─────────────────── canonical k-subgraph type ───────────────────────────────

def build_type_lookup(k):
    """Map EVERY 2^C(k,2) edge pattern -> canonical type id (by permutation).

    CRITICAL: must map ALL bit patterns (not just canonical ones) to their type,
    since induced subgraphs can appear in any vertex ordering.
    """
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    ne = len(edges)
    canon_to_type = {}
    type_id = 0
    arr = np.zeros(1 << ne, dtype=np.int32)
    for bits in range(1 << ne):
        A = np.zeros((k, k), dtype=np.int8)
        for idx, (i, j) in enumerate(edges):
            if (bits >> idx) & 1:
                A[i, j] = A[j, i] = 1
        # canonical = min over all k! permutations
        min_bits = min(
            sum(int(A[perm[i], perm[j]]) << eidx
                for eidx, (i, j) in enumerate(edges))
            for perm in permutations(range(k))
        )
        if min_bits not in canon_to_type:
            canon_to_type[min_bits] = type_id
            type_id += 1
        arr[bits] = canon_to_type[min_bits]  # map THIS bits -> its canonical type
    return arr, type_id


def count_induced_subs(adj, n, k, lookup, num_types):
    """Count induced k-subgraph types for a single graph."""
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    counts = np.zeros(num_types, dtype=np.int32)
    for sub in combinations(range(n), k):
        bits = 0
        for eidx, (li, lj) in enumerate(edges):
            if adj[sub[li], sub[lj]]:
                bits |= (1 << eidx)
        counts[lookup[bits]] += 1
    return tuple(counts)


def compute_sigs_all(graphs, n, k_list, lookups):
    """For each graph, compute combined signature over all k in k_list."""
    sigs = []
    for adj in graphs:
        sig = ()
        for k in k_list:
            lookup, num_types = lookups[k]
            sig += count_induced_subs(adj, n, k, lookup, num_types)
        sigs.append(sig)
    return sigs


# ─────────────────── graph loading ───────────────────────────────────────────

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


def load_g6(filepath):
    graphs = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                graphs.append(parse_graph6(line))
    return graphs


def get_n7_graphs():
    """Get all non-iso 7-vertex graphs via networkx atlas."""
    atlas = nx.graph_atlas_g()
    graphs = []
    for G in atlas:
        if G.number_of_nodes() == 7:
            graphs.append(nx.to_numpy_array(G, dtype=np.int8))
    return graphs


# ─────────────────── analysis ────────────────────────────────────────────────

def test_n(n, graphs, k_list):
    print(f"\n{'='*65}")
    print(f"  n={n}: {len(graphs)} non-iso graphs")
    print(f"{'='*65}")

    # Build lookups
    lookups = {}
    for k in k_list:
        t0 = time.time()
        lookup, num_types = build_type_lookup(k)
        print(f"  k={k}: {num_types} canonical types, lookup built in {time.time()-t0:.2f}s")
        lookups[k] = (lookup, num_types)

    cumulative_sig = [() for _ in graphs]

    for k in k_list:
        lookup, num_types = lookups[k]
        t0 = time.time()
        new_sigs = []
        for adj in graphs:
            new_sigs.append(count_induced_subs(adj, n, k, lookup, num_types))
        cumulative_sig = [cumulative_sig[i] + new_sigs[i] for i in range(len(graphs))]
        elapsed = time.time() - t0

        # Count collisions
        sig_groups = defaultdict(list)
        for i, s in enumerate(cumulative_sig):
            sig_groups[s].append(i)
        n_distinct = len(sig_groups)
        n_coll = sum(len(v) - 1 for v in sig_groups.values() if len(v) > 1)
        n_coll_groups = sum(1 for v in sig_groups.values() if len(v) > 1)

        print(f"\n  After k<={k}:  {n_distinct}/{len(graphs)} distinct  "
              f"({n_coll_groups} collision groups, {n_coll} extra)  [{elapsed:.1f}s]")

        if n_coll_groups == 0:
            print(f"  *** k<={k} = COMPLETE CLASSIFICATION for n={n}! ***")
            break
        else:
            # Show hard pairs
            hard = [(s, idxs) for s, idxs in sig_groups.items() if len(idxs) > 1]
            for s, idxs in hard[:3]:
                g1, g2 = graphs[idxs[0]], graphs[idxs[1]]
                e1 = int(g1.sum()) // 2
                e2 = int(g2.sum()) // 2
                d1 = tuple(sorted(g1.sum(axis=1).tolist()))
                print(f"    Hard pair {idxs[0]},{idxs[1]}: {e1},{e2} edges  deg={d1}")
            if len(hard) > 3:
                print(f"    ... and {len(hard)-3} more collision groups")


if __name__ == '__main__':
    # ── n=7 ──────────────────────────────────────────────────────────────────
    print("Loading n=7 graphs from networkx atlas...")
    t0 = time.time()
    g7 = get_n7_graphs()
    print(f"  Got {len(g7)} graphs in {time.time()-t0:.1f}s")
    test_n(7, g7, k_list=[3, 4, 5])

    # ── n=8 ──────────────────────────────────────────────────────────────────
    print(f"\nLoading n=8 graphs from {GRAPH8_FILE}...")
    t0 = time.time()
    g8 = load_g6(GRAPH8_FILE)
    print(f"  Got {len(g8)} graphs in {time.time()-t0:.1f}s")
    test_n(8, g8, k_list=[4, 5, 6])
