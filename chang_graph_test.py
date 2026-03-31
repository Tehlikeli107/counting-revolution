"""
Test induced k-subgraph counting on Chang graphs.
There are 4 non-isomorphic SRGs with parameters (28, 12, 6, 4):
  1. Triangular graph T(8) = J(8,2)
  2-4. Three Chang graphs (obtained by Seidel switching T(8))

If these have the same k-subgraph distributions, they're hard cases.
T(8) and the Chang graphs are the hardest known (28,12,6,4) SRG instances.
"""
import numpy as np
import networkx as nx
from itertools import combinations, permutations
from collections import Counter
import time


def make_j82():
    """Johnson graph J(8,2): vertices=2-subsets of [8], edges=intersection-size-1."""
    verts = list(combinations(range(8), 2))
    n = len(verts)  # 28
    v_to_idx = {v: i for i, v in enumerate(verts)}
    A = np.zeros((n, n), dtype=np.int8)
    for i, v in enumerate(verts):
        for j, w in enumerate(verts):
            if len(set(v) & set(w)) == 1:
                A[i, j] = 1
    return A, verts


def seidel_switch(A, S):
    """Seidel switching: flip adjacency between S and its complement."""
    n = len(A)
    B = A.copy()
    for i in S:
        for j in range(n):
            if j not in S:
                B[i, j] = 1 - A[i, j]
                B[j, i] = 1 - A[j, i]
    return B


def make_chang_graphs():
    """
    Construct all 4 (28,12,6,4) SRGs.
    T(8): triangular graph.
    Chang 1,2,3: obtained by specific Seidel switchings.
    Reference: Chang (1959,1960), Shrikhande, Bhagwandas
    """
    A_t8, verts = make_j82()
    v_to_idx = {v: i for i, v in enumerate(verts)}
    n = 28

    graphs = {'T8': A_t8}

    # Valid switching sets for SRG(28,12,6,4) Seidel switchings:
    # For |S|=4: need S independent, every v outside has exactly 2 neighbors in S
    #   -> Perfect matching of K8 works: {(0,1),(2,3),(4,5),(6,7)}
    # For |S|=8: need every v in S has exactly 2 neighbors in S, every v outside has 4
    #   -> S must be edge set of a 2-regular spanning subgraph H of K8
    #      i.e., H is a cycle cover of K8's 8 vertices: C8, C4+C4, or C3+C5

    # Chang1: switch on perfect matching (size 4) = {(0,1),(2,3),(4,5),(6,7)}
    M = [v_to_idx[(0,1)], v_to_idx[(2,3)], v_to_idx[(4,5)], v_to_idx[(6,7)]]
    chang1 = seidel_switch(A_t8, M)
    graphs['Chang1'] = chang1

    # Chang2: switch on Hamiltonian cycle C8 (size 8) = 0-1-2-3-4-5-6-7-0
    C8 = [v_to_idx[(0,1)], v_to_idx[(1,2)], v_to_idx[(2,3)], v_to_idx[(3,4)],
          v_to_idx[(4,5)], v_to_idx[(5,6)], v_to_idx[(6,7)], v_to_idx[(0,7)]]
    chang2 = seidel_switch(A_t8, C8)
    graphs['Chang2'] = chang2

    # Chang3: switch on C3+C5 (size 8) = (0-1-2-0) + (3-4-5-6-7-3)
    C3C5 = [v_to_idx[(0,1)], v_to_idx[(1,2)], v_to_idx[(0,2)],
            v_to_idx[(3,4)], v_to_idx[(4,5)], v_to_idx[(5,6)], v_to_idx[(6,7)], v_to_idx[(3,7)]]
    chang3 = seidel_switch(A_t8, C3C5)
    graphs['Chang3'] = chang3

    return graphs


def verify_srg(A, name):
    """Verify that A is a (28,12,6,4) SRG."""
    n = len(A)
    degs = A.sum(axis=1)
    if not np.all(degs == 12):
        return f"  {name}: NOT regular (degrees vary)"
    # Check lambda (triangles per edge)
    lams = []
    for i in range(n):
        for j in range(n):
            if A[i,j]:
                cn = int(A[i] @ A[j])
                lams.append(cn)
    # Check mu (paths-2 between non-adjacent)
    mus = []
    for i in range(n):
        for j in range(n):
            if not A[i,j] and i!=j:
                cn = int(A[i] @ A[j])
                mus.append(cn)
    if len(set(lams))==1 and len(set(mus))==1:
        return f"  {name}: SRG(28,12,{lams[0]},{mus[0]}) [VALID]"
    else:
        return f"  {name}: NOT SRG (lambda={set(lams)}, mu={set(mus)})"


def build_type_lookup(k):
    """Maps ALL bit patterns to canonical type (correct version)."""
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
    return arr, type_id


def compute_sig(adj, n, k, lookup, num_types):
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    counts = np.zeros(num_types, dtype=np.int32)
    for sub in combinations(range(n), k):
        bits = 0
        for eidx, (li, lj) in enumerate(edges):
            if adj[sub[li], sub[lj]]:
                bits |= (1 << eidx)
        counts[lookup[bits]] += 1
    return tuple(counts)


if __name__ == '__main__':
    print("Chang Graph Test: (28,12,6,4) SRGs")
    print("=" * 60)

    print("\nConstructing graphs...")
    graphs = make_chang_graphs()

    print("\nVerifying SRG properties:")
    for name, A in graphs.items():
        G = nx.from_numpy_array(A.astype(int))
        print(verify_srg(A, name))

    print("\nChecking non-isomorphism:")
    names = list(graphs.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            Gi = nx.from_numpy_array(graphs[names[i]].astype(int))
            Gj = nx.from_numpy_array(graphs[names[j]].astype(int))
            iso = nx.is_isomorphic(Gi, Gj)
            print(f"  {names[i]} iso {names[j]}? {iso}")

    n = 28
    for k in range(4, 9):
        print(f"\nBuilding k={k} type lookup...", end='', flush=True)
        t0 = time.time()
        lookup, num_types = build_type_lookup(k)
        print(f" {num_types} types, {time.time()-t0:.1f}s")

        sigs = {}
        for name, A in graphs.items():
            t0 = time.time()
            sig = compute_sig(A, n, k, lookup, num_types)
            sigs[name] = sig

        # Check collisions
        sig_vals = list(sigs.values())
        sig_names = list(sigs.keys())
        collisions = []
        for i in range(len(sig_names)):
            for j in range(i+1, len(sig_names)):
                if sigs[sig_names[i]] == sigs[sig_names[j]]:
                    collisions.append((sig_names[i], sig_names[j]))

        if collisions:
            print(f"  k={k}: {len(collisions)} collision(s): {collisions}")
        else:
            print(f"  k={k}: ALL 4 DISTINCT! Chang graphs distinguished at k={k}")
            break
