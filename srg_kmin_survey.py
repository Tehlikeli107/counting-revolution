"""
Survey of minimum k to distinguish known non-isomorphic SRG pairs.
Tests all known "hard" SRG families where multiple non-iso examples exist.
"""
import numpy as np
import networkx as nx
from itertools import combinations, permutations
import time


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


def find_kmin(graphs, names, max_k=10):
    """Find minimum k to distinguish all graphs."""
    n = len(graphs[0])
    for k in range(3, max_k + 1):
        t0 = time.time()
        lookup, num_types = build_type_lookup(k)
        sigs = [compute_sig(g, n, k, lookup, num_types) for g in graphs]
        elapsed = time.time() - t0
        all_distinct = len(set(sigs)) == len(sigs)
        collisions = [(names[i], names[j])
                      for i in range(len(sigs))
                      for j in range(i+1, len(sigs))
                      if sigs[i] == sigs[j]]
        if all_distinct:
            return k, elapsed, []
        if elapsed > 60:
            return None, elapsed, collisions
    return None, 0, []


# ================================================================
# SRG CONSTRUCTIONS
# ================================================================

def make_petersen():
    """Petersen graph SRG(10,3,0,1) — unique."""
    G = nx.petersen_graph()
    return nx.to_numpy_array(G, dtype=np.int8)


def make_rook44():
    """Rook(4,4) SRG(16,6,2,2)."""
    verts = [(i,j) for i in range(4) for j in range(4)]
    n = len(verts)
    A = np.zeros((n,n), dtype=np.int8)
    for i,(r1,c1) in enumerate(verts):
        for j,(r2,c2) in enumerate(verts):
            if i!=j and (r1==r2 or c1==c2):
                A[i,j] = 1
    return A


def make_shrikhande():
    """Shrikhande SRG(16,6,2,2)."""
    verts = [(i,j) for i in range(4) for j in range(4)]
    n = len(verts)
    diffs = {(1,0),(3,0),(0,1),(0,3),(1,1),(3,3)}
    A = np.zeros((n,n), dtype=np.int8)
    for i,(a,b) in enumerate(verts):
        for j,(c,d) in enumerate(verts):
            if i!=j and ((a-c)%4,(b-d)%4) in diffs:
                A[i,j] = 1
    return A


def make_paley17():
    """Paley graph on GF(17): SRG(17,8,3,4)."""
    n = 17
    # Quadratic residues mod 17
    qr = {x*x % 17 for x in range(1,17)}  # {1,2,4,8,9,13,15,16}
    A = np.zeros((n,n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i != j and (i-j) % 17 in qr:
                A[i,j] = 1
    return A


def make_paley13():
    """Paley graph on GF(13): SRG(13,6,2,3)."""
    n = 13
    qr = {x*x % 13 for x in range(1,13)}
    A = np.zeros((n,n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i != j and (i-j) % 13 in qr:
                A[i,j] = 1
    return A


def make_J82():
    """Triangular graph T(8) = J(8,2): SRG(28,12,6,4)."""
    verts = list(combinations(range(8), 2))
    n = len(verts)
    v_to_idx = {v: i for i,v in enumerate(verts)}
    A = np.zeros((n,n), dtype=np.int8)
    for i,v in enumerate(verts):
        for j,w in enumerate(verts):
            if len(set(v)&set(w)) == 1:
                A[i,j] = 1
    return A, v_to_idx, verts


def seidel_switch(A, S):
    n = len(A)
    B = A.copy()
    for i in S:
        for j in range(n):
            if j not in S:
                B[i,j] = 1 - A[i,j]
                B[j,i] = 1 - A[j,i]
    return B


def make_chang_graphs():
    A_t8, v_to_idx, verts = make_J82()
    # Chang1: perfect matching {(0,1),(2,3),(4,5),(6,7)}
    M = [v_to_idx[(0,1)], v_to_idx[(2,3)], v_to_idx[(4,5)], v_to_idx[(6,7)]]
    # Chang2: Hamiltonian cycle 0-1-2-3-4-5-6-7-0
    C8 = [v_to_idx[(0,1)], v_to_idx[(1,2)], v_to_idx[(2,3)], v_to_idx[(3,4)],
          v_to_idx[(4,5)], v_to_idx[(5,6)], v_to_idx[(6,7)], v_to_idx[(0,7)]]
    # Chang3: C3+C5
    C3C5 = [v_to_idx[(0,1)], v_to_idx[(1,2)], v_to_idx[(0,2)],
            v_to_idx[(3,4)], v_to_idx[(4,5)], v_to_idx[(5,6)],
            v_to_idx[(6,7)], v_to_idx[(3,7)]]
    return [A_t8,
            seidel_switch(A_t8, M),
            seidel_switch(A_t8, C8),
            seidel_switch(A_t8, C3C5)]


def make_paley25():
    """Paley graph on GF(25)=GF(5^2): SRG(25,12,5,6)."""
    # GF(25) as Z5[x]/(x^2+x+1) where x^2=-x-1 mod 5: x^2+x+1 is irreducible over Z5
    # Elements as pairs (a,b) = a + b*alpha
    from itertools import product
    elems = [(a,b) for a in range(5) for b in range(5)]  # 25 elements
    elem_idx = {e: i for i,e in enumerate(elems)}

    def mul(u, v):
        a,b = u; c,d = v
        # (a+bx)(c+dx) = ac + (ad+bc)x + bdx^2 = ac + (ad+bc)x + bd(-x-1)
        # = (ac-bd) + (ad+bc-bd)x mod 5
        return ((a*c - b*d) % 5, (a*d + b*c - b*d) % 5)

    def sq(u):
        return mul(u, u)

    # Compute all non-zero squares
    nonzero = [e for e in elems if e != (0,0)]
    squares = set()
    for e in nonzero:
        s = sq(e)
        if s != (0,0):
            squares.add(s)

    # Check it's a valid QR set of size 12
    # In GF(q) with q=1 mod 4, QRs form a set of size (q-1)/2 = 12

    def sub(u, v):
        return ((u[0]-v[0])%5, (u[1]-v[1])%5)

    n = 25
    A = np.zeros((n,n), dtype=np.int8)
    for i,u in enumerate(elems):
        for j,v in enumerate(elems):
            if i != j and sub(u,v) in squares:
                A[i,j] = 1
    return A


def make_hoffmansingleton():
    """Hoffman-Singleton graph SRG(50,7,0,1) — unique."""
    G = nx.hoffman_singleton_graph()
    return nx.to_numpy_array(G, dtype=np.int8)


# ================================================================
# SURVEY
# ================================================================

if __name__ == '__main__':
    print("SRG k_min Survey")
    print("=" * 65)
    print(f"{'Graph family':<35} {'n':>4} {'params':<18} {'k_min':>6}")
    print("-" * 65)

    # 1. Shrikhande vs Rook(4,4) — WL-1 fails
    graphs = [make_rook44(), make_shrikhande()]
    names = ['Rook(4,4)', 'Shrikhande']
    k, t, colls = find_kmin(graphs, names, max_k=8)
    print(f"  {'Shrikhande/Rook(4,4) pair':<33} {16:>4} {'SRG(16,6,2,2)':<18} {k if k else '?':>6}  ({t:.1f}s)")

    # 2. Chang graphs (4 non-iso SRGs)
    graphs = make_chang_graphs()
    names = ['T(8)', 'Chang1', 'Chang2', 'Chang3']
    k, t, colls = find_kmin(graphs, names, max_k=9)
    print(f"  {'Chang graphs (4 SRGs)':<33} {28:>4} {'SRG(28,12,6,4)':<18} {k if k else '?':>6}  ({t:.1f}s)")

    # 3. Paley 13 — unique
    A = make_paley13()
    print(f"  {'Paley(13) — unique SRG':<33} {13:>4} {'SRG(13,6,2,3)':<18} {'unique':>6}")

    # 4. Paley 17 — unique (Paley(17) is unique SRG(17,8,3,4))
    A = make_paley17()
    print(f"  {'Paley(17) — unique SRG':<33} {17:>4} {'SRG(17,8,3,4)':<18} {'unique':>6}")

    # 5. Paley(25) vs other SRG(25,12,5,6)
    # There are 15 SRGs with params (25,12,5,6), including Paley(25) and Paley(5)^2 etc.
    # Let's check if we can build two distinct ones and test
    # The complement of SRG(25,12,5,6) is SRG(25,12,5,6) (self-complementary params)
    A_p25 = make_paley25()
    print(f"  {'Paley(25): SRG(25,12,5,6)':<33} {25:>4} {'SRG(25,12,5,6)':<18} {'(1 of 15)':>6}")

    # 6. Hoffman-Singleton — unique SRG(50,7,0,1)
    A_hs = make_hoffmansingleton()
    print(f"  {'Hoffman-Singleton':<33} {50:>4} {'SRG(50,7,0,1)':<18} {'unique':>6}")

    print()
    print("Summary:")
    print("  Shrikhande/Rook(4,4): WL-1 fails, our method k=?")
    print("  Chang graphs: 4 non-iso SRGs(28), our method k=?")
    print("  Paley/H-S: unique for their params (trivially classified)")
