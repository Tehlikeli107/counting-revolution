"""
Shrikhande vs Rook(4,4): SRG(16,6,2,2) pair
WL-1 FAILS to distinguish these.
Test: what minimum k of induced subgraph counting distinguishes them?
"""
import numpy as np
import networkx as nx
from itertools import combinations, permutations
from collections import Counter
import time


def make_rook44():
    """Rook graph L2(4) = K4 x K4 Cartesian product.
    Vertices: (i,j) for i,j in Z4. Adjacent iff share row or column."""
    verts = [(i, j) for i in range(4) for j in range(4)]
    n = len(verts)  # 16
    v_to_idx = {v: k for k, v in enumerate(verts)}
    A = np.zeros((n, n), dtype=np.int8)
    for i, (r1, c1) in enumerate(verts):
        for j, (r2, c2) in enumerate(verts):
            if i != j and (r1 == r2 or c1 == c2):
                A[i, j] = 1
    return A


def make_shrikhande():
    """Shrikhande graph: vertices Z4 x Z4.
    (a,b) ~ (c,d) iff (a-c, b-d) in {(+/-1,0),(0,+/-1),(+1,+1),(-1,-1)} mod 4."""
    verts = [(i, j) for i in range(4) for j in range(4)]
    n = len(verts)
    v_to_idx = {v: k for k, v in enumerate(verts)}
    diffs = {(1,0),(3,0),(0,1),(0,3),(1,1),(3,3)}  # Z4 arithmetic: 3=-1 mod 4
    A = np.zeros((n, n), dtype=np.int8)
    for i, (a, b) in enumerate(verts):
        for j, (c, d) in enumerate(verts):
            if i != j and ((a-c) % 4, (b-d) % 4) in diffs:
                A[i, j] = 1
    return A


def verify_srg(A, name):
    n = len(A)
    degs = A.sum(axis=1)
    if not np.all(degs == degs[0]):
        return f"  {name}: NOT regular"
    d = int(degs[0])
    lams = [int(A[i] @ A[j]) for i in range(n) for j in range(n) if A[i,j]]
    mus = [int(A[i] @ A[j]) for i in range(n) for j in range(n) if not A[i,j] and i!=j]
    if len(set(lams)) == 1 and len(set(mus)) == 1:
        return f"  {name}: SRG({n},{d},{lams[0]},{mus[0]}) [VALID]"
    return f"  {name}: NOT SRG lam={set(lams)} mu={set(mus)}"


def build_type_lookup(k):
    """Maps ALL bit patterns to canonical type."""
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
    print("Shrikhande vs Rook(4,4): SRG(16,6,2,2)")
    print("=" * 50)

    A_rook = make_rook44()
    A_shri = make_shrikhande()

    print("\nVerifying SRG properties:")
    print(verify_srg(A_rook, "Rook(4,4)"))
    print(verify_srg(A_shri, "Shrikhande"))

    G_r = nx.from_numpy_array(A_rook.astype(int))
    G_s = nx.from_numpy_array(A_shri.astype(int))
    print(f"\nIsomorphic? {nx.is_isomorphic(G_r, G_s)}")
    print(f"Rook eigs: {sorted(np.linalg.eigvalsh(A_rook.astype(float)))[:5]}...")
    print(f"Shri eigs: {sorted(np.linalg.eigvalsh(A_shri.astype(float)))[:5]}...")

    n = 16
    print(f"\nTesting k=3..{n//2}:")
    for k in range(3, n // 2 + 1):
        t0 = time.time()
        lookup, num_types = build_type_lookup(k)
        sig_r = compute_sig(A_rook, n, k, lookup, num_types)
        sig_s = compute_sig(A_shri, n, k, lookup, num_types)
        elapsed = time.time() - t0
        same = (sig_r == sig_s)
        print(f"  k={k}: {'COLLISION' if same else 'DISTINCT'} ({num_types} types, {elapsed:.1f}s)")
        if not same:
            # Show which types differ
            diffs = [(i, sig_r[i], sig_s[i]) for i in range(num_types) if sig_r[i] != sig_s[i]]
            print(f"    {len(diffs)} type counts differ (first 3): {diffs[:3]}")
            break
