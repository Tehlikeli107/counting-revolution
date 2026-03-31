"""
CFI Graphs vs Counting Revolution
===================================
The Cai-Furer-Immerman (1992) construction creates pairs of non-isomorphic
graphs indistinguishable by ALL k-WL algorithms.

Question: Do our 12 counting invariants (K3,K4,K5, induced subgraph
distribution, etc.) also fail on CFI pairs?

If YES -> expected, CFI was designed to defeat counting invariants
If NO  -> our invariants are STRONGER than predicted theory

CFI construction on a graph G:
  For each vertex v of G with degree d(v):
    Create 2 "copies" of v: one for bit 0, one for bit 1
    Create C(d(v),2) "edge vertices" connecting them
  For each edge (u,v) of G:
    Create an "XOR gadget" linking the edge endpoints

Standard simple CFI:
  For each vertex v of degree d, create 2^d gadget vertices.
  For K3 (all vertices degree 2): 3 vertices * 4 gadgets = 12 vertices.

  "Twisted" CFI: flip one vertex gadget -> non-isomorphic but same counts.
"""

import numpy as np
from itertools import combinations
from collections import Counter


# ============================================================
# CFI CONSTRUCTION
# ============================================================

def build_cfi_graph(G_edges, G_vertices, twisted_vertices=set()):
    """
    Build CFI graph based on connected graph G.

    Encoding:
    For each vertex v of G with neighbors u1, ..., ud:
      Create 2^d gadget vertices: v(b1,...,bd) for each (b1,...,bd) in {0,1}^d
      Two gadget vertices v(b...) and v(b'...) are connected iff b and b' differ
      in EXACTLY ONE coordinate AND that coordinate flip comes from an edge to some u_i.

    For each edge (u,v) of G:
      Connect gadget vertex v(b...) to gadget u(b'...) iff
      the bits corresponding to the edge (u,v) are EQUAL (parity matching).
      If v is twisted: flip the parity for vertex v.

    Simplified version (for K3, d=2):
    For vertex v with neighbors [u, w]:
      4 gadgets: v00, v01, v10, v11
      Internal edges: v00-v11 (differ in both), v01-v10 (differ in both)
      Wait, INTERNAL edges connect vertices differing in exactly one bit -- no that's a hypercube.

    Actually, use the standard "or-gadget" CFI:
    For each vertex v of degree d: create 2 parts of 2^{d-1} vertices each.
    Vertices in same part are connected to each other (complete).
    Vertices in different parts connected based on XOR.

    Simple version for K3 (Evdokimov-Ponomarenko CFI for K3):
    Each edge of K3 becomes a "wire" with 2 ports.
    Each vertex becomes a "gate" connecting its d=2 wires.
    """

    # Use the "canonical" small CFI on K_3 from CFI 1992
    # K3: vertices {0,1,2}, edges {(0,1),(1,2),(0,2)}
    # Each vertex v has neighbors; sort neighbors to define wire order

    # For K3: each vertex has degree 2
    # For each vertex v: create 2 "half-vertices": v_0 and v_1
    # For each edge (u,v): create 4 "edge gadget" vertices: (uv_00, uv_01, uv_10, uv_11)
    #
    # This gives 3*2 + 3*4 = 6 + 12 = 18 vertex gadgets? That's a lot.
    #
    # Let me use a simpler version: the "XOR construction" from Babai-Cai-Immerman 1990:
    # For graph G on n vertices:
    #   For each vertex v: 2 copies (v^0, v^1)
    #   For each edge (u,v): 2 edge gadgets (eu,ev) -> creates 4 vertices from each edge
    # But this gives 2n + 4m vertices for K3: 6 + 12 = 18 vertices

    # Let me use Köbler-Schöning-Torán's simplified CFI:
    # G = (V, E); for each v in V and each bit pattern of deg(v) bits:
    # create one vertex. Connect two vertex-gadgets if they differ in XOR = 1 via one edge.

    # For K3: deg=2 for all vertices. 2^2 = 4 gadgets per vertex. 4*3 = 12 vertices total.

    n = len(G_vertices)
    # Adjacency list
    adj = {v: [] for v in G_vertices}
    for (u, v) in G_edges:
        adj[u].append(v)
        adj[v].append(u)
    # Sort neighbors for canonical ordering
    adj = {v: sorted(adj[v]) for v in adj}

    # For each vertex v with degree d = len(adj[v]):
    # Create 2^d gadget vertices encoded as (v, tuple_of_bits)
    # OR (for simplicity): just 2 gadgets per vertex: (v, 0) and (v, 1)
    # where the bit represents the "parity" of the XOR of all bits

    # SIMPLE CFI VARIANT (Fürer's simplified version):
    # Each vertex v has 2 copies: v_0, v_1
    # For each edge (u,v): connect u_0--v_0 and u_1--v_1 (same parity)
    #                  AND  u_0--v_1 and u_1--v_0 (cross parity)
    # Wait, that would make K4 on every edge...

    # Let me just use the EXPLICIT construction from the literature for small cases.
    # For K3 with vertices {a, b, c}:
    #
    # Standard CFI:
    # Vertex gadgets: a_0, a_1, b_0, b_1, c_0, c_1 (2 per vertex = 6 vertices)
    # Edge gadgets for (a,b): 4 vertices: ab_00, ab_01, ab_10, ab_11
    #                        connected to a_0/a_1 and b_0/b_1 based on parity
    # Edge gadgets for (b,c): 4 vertices: bc_00, bc_01, bc_10, bc_11
    # Edge gadgets for (a,c): 4 vertices: ac_00, ac_01, ac_10, ac_11
    # Total: 6 + 12 = 18 vertices

    # Actually, let me find the canonical CFI(K3) in literature.
    # From Dawar et al.: CFI(K3) has 6 vertices per original vertex, so 18 total.
    # No wait...

    # I'll use the formulation from Schweitzer:
    # For each vertex v, create |N(v)| "port" vertices plus 1 "center" vertex.
    # For K3 (degree 2): 1 center + 2 ports = 3 vertices per original vertex = 9 total
    # Plus edge connections.

    # OK let me just implement the XOR/parity gadget construction directly.
    pass


def build_cfi_k3():
    """
    Build the canonical CFI pair on K3.

    K3 has 3 vertices {0,1,2} and 3 edges {01,12,02}.

    CFI construction (following Immerman-Lander):
    For each edge e=(u,v): create 2 "half-edges" h_e^u (left end) and h_e^v (right end).
    For each vertex v with edges e1, e2 (degree 2):
      Create edges: h_e1^v -- h_e2^v (cross-connection)
      Optionally "twist": XOR the connection

    Result: two graphs on 12 vertices (2 half-edges * 3 edges = 6 + original K3 = 9?)

    Let me just hard-code the CFI(K3) adjacency matrix from a reference.
    """

    # From the paper "Cai-Furer-Immerman Graphs" — standard construction
    # CFI(K3): 12 vertices, defined as follows
    #
    # Vertices: for each edge of K3, create 2 vertices: "near" the edge's endpoints
    # For edge (0,1): vertices 0a, 0b (near 0), 1a, 1b (near 1)
    # For edge (1,2): vertices 1c, 1d (near 1), 2a, 2b (near 2)
    # For edge (0,2): vertices 0c, 0d (near 0), 2c, 2d (near 2)
    #
    # Wait this gives 12 vertices. And the encoding is complex.
    # Let me use a different approach: symmetric difference/XOR gadget.

    # SIMPLEST FORMULATION:
    # K3 vertices: a, b, c
    # For each vertex, 4 gadget vertices (indexed by 2-bit strings 00,01,10,11)
    # Vertex a's gadgets: a00, a01, a10, a11  (indexed by (bit for a-b edge, bit for a-c edge))
    # Vertex b's gadgets: b00, b01, b10, b11  (indexed by (bit for a-b edge, bit for b-c edge))
    # Vertex c's gadgets: c00, c01, c10, c11  (indexed by (bit for a-c edge, bit for b-c edge))
    #
    # Gadget-internal edges: connect two gadgets of same vertex if they differ in EXACTLY ONE bit
    # (This makes each vertex gadget a 2-dimensional hypercube = C4)
    #
    # Cross-edges: connect a_{b1, c1} to b_{a1, c2} iff a1 == b1 (same bit for shared edge a-b)
    # Cross-edges: connect a_{b1, c1} to c_{a1, b2} iff c1 == a1 (same bit for shared edge a-c)
    # Cross-edges: connect b_{a1, c1} to c_{b1, a2} iff c1 == b1 (same bit for shared edge b-c)
    #
    # TWIST: for one vertex (say a), flip all cross-edge parities
    # (connect iff a1 != b1 for shared edges involving a)

    # Vertex indices:
    # a: 00->0, 01->1, 10->2, 11->3
    # b: 00->4, 01->5, 10->6, 11->7
    # c: 00->8, 01->9, 10->10, 11->11

    def gadget_idx(vertex, bits):
        """vertex in {a=0, b=1, c=2}, bits is 2-tuple of ints."""
        return vertex * 4 + bits[0] * 2 + bits[1]

    def build_adj(twisted=False):
        N = 12
        A = np.zeros((N, N), dtype=np.int32)

        def add_edge(u, v):
            A[u, v] = A[v, u] = 1

        # Internal edges within each vertex gadget (hypercube C4 edges)
        for vert in range(3):
            # Connect (b0,b1) to (1-b0, b1) and (b0, 1-b1) — Hamming distance 1
            for b0 in range(2):
                for b1 in range(2):
                    # Flip b0
                    add_edge(gadget_idx(vert, (b0,b1)), gadget_idx(vert, (1-b0, b1)))
                    # Flip b1
                    add_edge(gadget_idx(vert, (b0,b1)), gadget_idx(vert, (b0, 1-b1)))

        # Cross-edges: bit assignment
        # a's bits: (bit_ab, bit_ac)  -> vertex 0
        # b's bits: (bit_ab, bit_bc)  -> vertex 1
        # c's bits: (bit_ac, bit_bc)  -> vertex 2
        #
        # Edge a-b: connect a(bit_ab=x, *) to b(bit_ab=x, *) for all wildcard values
        # i.e., a(x, y) connects to b(x, z) for all y, z
        #
        # This gives a "complete bipartite" connection between the halves of a and b
        # with the SAME value of bit_ab.

        # a-b edge: a's first bit (bit_ab) must match b's first bit (bit_ab)
        # Normal: a(x,y) -- b(x,z) for all y,z; i.e., same x
        # Twisted (twist on a): a(x,y) -- b(1-x,z) for all y,z (flip parity)

        for y in range(2):  # a's second bit (bit_ac)
            for z in range(2):  # b's second bit (bit_bc)
                for x in range(2):  # shared bit
                    a_node = gadget_idx(0, (x, y))     # a(bit_ab=x, bit_ac=y)
                    if twisted:
                        b_node = gadget_idx(1, (1-x, z))  # twisted: flip parity
                    else:
                        b_node = gadget_idx(1, (x, z))    # normal: same parity
                    add_edge(a_node, b_node)

        # a-c edge: a's second bit (bit_ac) must match c's first bit (bit_ac)
        # Normal: a(x,y) -- c(y,z) for all x, z
        for x in range(2):  # a's first bit (bit_ab)
            for z in range(2):  # c's second bit (bit_bc)
                for y in range(2):  # shared bit
                    a_node = gadget_idx(0, (x, y))     # a(bit_ab=x, bit_ac=y)
                    c_node = gadget_idx(2, (y, z))     # c(bit_ac=y, bit_bc=z)
                    add_edge(a_node, c_node)

        # b-c edge: b's second bit (bit_bc) must match c's second bit (bit_bc)
        # Normal: b(x,y) -- c(z,y) for all x, z
        for x in range(2):  # b's first bit (bit_ab)
            for z in range(2):  # c's first bit (bit_ac)
                for y in range(2):  # shared bit
                    b_node = gadget_idx(1, (x, y))     # b(bit_ab=x, bit_bc=y)
                    c_node = gadget_idx(2, (z, y))     # c(bit_ac=z, bit_bc=y)
                    add_edge(b_node, c_node)

        np.fill_diagonal(A, 0)
        return A

    A_normal  = build_adj(twisted=False)
    A_twisted = build_adj(twisted=True)
    return A_normal, A_twisted


# ============================================================
# COUNTING INVARIANTS
# ============================================================

def count_cliques(A, k):
    n = len(A)
    return sum(1 for verts in combinations(range(n), k)
               if all(A[verts[i],verts[j]]==1 for i in range(k) for j in range(i+1,k)))

def degree_sequence(A):
    return tuple(sorted(A.sum(0).tolist()))

def eigenvalues(A):
    import numpy.linalg as la
    eigs = la.eigvalsh(A.astype(float))
    return tuple(round(e, 3) for e in sorted(eigs))

def induced_subgraph_dist(A, k):
    n = len(A)
    dist = Counter()
    for verts in combinations(range(n), k):
        key = tuple(A[verts[i],verts[j]] for i in range(k) for j in range(i+1,k))
        dist[key] += 1
    return dist

def wl_1(A, max_iters=50):
    n = len(A)
    colors = tuple(int(x) for x in A.sum(1).tolist())
    for _ in range(max_iters):
        nc = [tuple(sorted([colors[u] for u in range(n) if A[v,u]])) for v in range(n)]
        nc_tuple = tuple(nc)
        if nc_tuple == colors: break
        cm = {}
        nid = [0]
        def gid(c):
            if c not in cm: cm[c] = nid[0]; nid[0] += 1
            return cm[c]
        colors = tuple(gid(c) for c in nc_tuple)
    return tuple(sorted(colors))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*65)
    print("CFI GRAPHS vs COUNTING REVOLUTION")
    print("  CFI(K3): canonical quantum isomorphic graph pair on 12 vertices")
    print("="*65)
    print()

    A_norm, A_twist = build_cfi_k3()

    n1, n2 = len(A_norm), len(A_twist)
    e1 = A_norm.sum() // 2
    e2 = A_twist.sum() // 2
    print(f"CFI_normal:  {n1} vertices, {e1} edges")
    print(f"CFI_twisted: {n2} vertices, {e2} edges")
    print()

    # Verify they're non-isomorphic (different permutation classes)
    # Quick check: sorted degree sequences and spectra
    d1 = degree_sequence(A_norm)
    d2 = degree_sequence(A_twist)
    print(f"Degree sequences identical: {d1 == d2}")
    if d1 != d2:
        print(f"  Normal:  {d1}")
        print(f"  Twisted: {d2}")

    eig1 = eigenvalues(A_norm)
    eig2 = eigenvalues(A_twist)
    print(f"Eigenvalues identical: {eig1 == eig2}")

    w1_n = wl_1(A_norm)
    w1_t = wl_1(A_twist)
    print(f"1-WL coloring identical: {w1_n == w1_t}")
    print()

    print("--- COUNTING INVARIANTS ---")
    print()

    for label, A in [("CFI_normal", A_norm), ("CFI_twisted", A_twist)]:
        k3 = count_cliques(A, 3)
        k4 = count_cliques(A, 4)
        k5 = count_cliques(A, 5)
        print(f"{label}:")
        print(f"  K3={k3}, K4={k4}, K5={k5}")

    print()
    k3_same = count_cliques(A_norm, 3) == count_cliques(A_twist, 3)
    k4_same = count_cliques(A_norm, 4) == count_cliques(A_twist, 4)
    print(f"K3 same: {k3_same}")
    print(f"K4 same: {k4_same}")

    print()
    print("--- INDUCED 3-SUBGRAPH DISTRIBUTION ---")
    d3_norm  = induced_subgraph_dist(A_norm, 3)
    d3_twist = induced_subgraph_dist(A_twist, 3)
    d3_same  = d3_norm == d3_twist
    print(f"3-subgraph dist identical: {d3_same}")

    print()
    print("--- INDUCED 4-SUBGRAPH DISTRIBUTION ---")
    d4_norm  = induced_subgraph_dist(A_norm, 4)
    d4_twist = induced_subgraph_dist(A_twist, 4)
    d4_same  = d4_norm == d4_twist
    print(f"4-subgraph dist identical: {d4_same}")

    print()
    print("="*65)
    print("SUMMARY")
    print("="*65)
    all_same = k3_same and k4_same and d3_same and d4_same
    if all_same:
        print("ALL counting invariants IDENTICAL -> CFI pair is counting-revolution-indistinguishable")
        print("This confirms: our counting invariants fail on CFI-type quantum isomorphic pairs.")
        print("The n=9 barrier (Paley(9)=Rook(3)) was a MISTAKE, but CFI PAIRS are genuine barriers.")
    else:
        inv_name = 'K4' if not k4_same else ('4-subgraph dist' if not d4_same else 'K3/3-subgraph')
        print(f"DISTINGUISHED by {inv_name}!")
        print("SURPRISING: Our counting invariants BEAT theoretical prediction!")
        print("CFI graphs should be indistinguishable by all counting invariants.")
        print("This means our CFI construction has an error, OR our invariants are stronger than theory predicts.")
