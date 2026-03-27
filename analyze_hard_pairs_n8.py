"""
Analyze the 4 hard pairs for n=8 (k=5 fails, k=6 resolves).
Find what 6-vertex induced subgraph types differ between them.
"""
import numpy as np
from itertools import combinations, permutations
from collections import defaultdict
import networkx as nx
import os

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


def build_type_lookup(k):
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
        min_bits = min(
            sum(int(A[perm[i], perm[j]]) << eidx
                for eidx, (i, j) in enumerate(edges))
            for perm in permutations(range(k))
        )
        if min_bits not in canon_to_type:
            canon_to_type[min_bits] = type_id
            type_id += 1
        arr[bits] = canon_to_type[min_bits]
    return arr, type_id, {v: k for k, v in canon_to_type.items()}  # type->canonical bits


def count_induced_subs(adj, n, k, lookup, num_types):
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    counts = np.zeros(num_types, dtype=np.int32)
    for sub in combinations(range(n), k):
        bits = 0
        for eidx, (li, lj) in enumerate(edges):
            if adj[sub[li], sub[lj]]:
                bits |= (1 << eidx)
        counts[lookup[bits]] += 1
    return counts


def print_graph_info(adj, name, n=8):
    G = nx.from_numpy_array(adj.astype(int))
    deg = sorted(adj.sum(axis=1).astype(int).tolist())
    e = int(adj.sum()) // 2
    conn = nx.is_connected(G)
    spec = sorted(np.linalg.eigvalsh(adj.astype(float)).tolist(), reverse=True)
    spec_str = [f"{x:.3f}" for x in spec]
    print(f"  {name}: {e} edges, deg={deg}, connected={conn}")
    print(f"    eigenvalues: [{', '.join(spec_str)}]")
    # Check if complement is isomorphic
    comp = 1 - adj - np.eye(n, dtype=np.int8)
    if nx.is_isomorphic(G, nx.from_numpy_array(comp.astype(int))):
        print(f"    SELF-COMPLEMENTARY!")


def adj_to_graph6(adj):
    """Convert adj matrix to human-readable edge list."""
    n = len(adj)
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j]]
    return edges


if __name__ == '__main__':
    print("Loading n=8 graphs...")
    with open(os.path.join(DATA_DIR, 'graph8.g6'), 'r') as f:
        all_lines = [l.strip() for l in f if l.strip() and not l.startswith('>')]
    graphs = [parse_graph6(l) for l in all_lines]
    print(f"  Loaded {len(graphs)} graphs\n")

    # The 4 hard pairs (from verify_kmin output)
    hard_pairs = [(3630, 4580), (7163, 7210), (7638, 8901), (11754, 11839)]

    print("Building type lookups...")
    lookup5, num5, id2bits5 = build_type_lookup(5)
    lookup6, num6, id2bits6 = build_type_lookup(6)
    print(f"  k=5: {num5} types, k=6: {num6} types\n")

    for i, (idx1, idx2) in enumerate(hard_pairs):
        g1, g2 = graphs[idx1], graphs[idx2]
        n = 8

        print(f"{'='*60}")
        print(f"Hard pair {i+1}: graphs {idx1} and {idx2}")
        print_graph_info(g1, f"G{idx1}", n)
        print_graph_info(g2, f"G{idx2}", n)

        # Verify k=5 same, k=6 different
        sig5_1 = count_induced_subs(g1, n, 5, lookup5, num5)
        sig5_2 = count_induced_subs(g2, n, 5, lookup5, num5)
        assert np.array_equal(sig5_1, sig5_2), "k=5 should match!"

        sig6_1 = count_induced_subs(g1, n, 6, lookup6, num6)
        sig6_2 = count_induced_subs(g2, n, 6, lookup6, num6)

        diff = sig6_1 - sig6_2
        diff_types = [(t, int(diff[t])) for t in range(num6) if diff[t] != 0]

        print(f"\n  k=5: SAME (confirmed)")
        print(f"  k=6: {len(diff_types)} differing types:")
        for t, d in diff_types:
            cbits = id2bits6[t]
            # Decode canonical bits back to edge pattern
            edges6 = [(i, j) for i in range(6) for j in range(i+1, 6)]
            A6 = np.zeros((6,6), dtype=int)
            for eidx, (ei, ej) in enumerate(edges6):
                if (cbits >> eidx) & 1:
                    A6[ei, ej] = A6[ej, ei] = 1
            deg6 = sorted(A6.sum(axis=1).tolist())
            e6 = int(A6.sum()) // 2
            sign = '+' if d > 0 else ''
            print(f"    type {t:3d} (edges={e6}, deg={deg6}): G{idx1} {sign}{d:+d} more")

        # Check isomorphism
        G1 = nx.from_numpy_array(g1.astype(int))
        G2 = nx.from_numpy_array(g2.astype(int))
        assert not nx.is_isomorphic(G1, G2), "Should NOT be isomorphic!"
        print(f"\n  Non-isomorphic: confirmed")
        print()
