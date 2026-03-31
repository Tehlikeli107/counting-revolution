"""
Analyze the 22 hard pairs for n=10 (k=6 fails, k=7 resolves).
From verify_kmin_n10_v2.py output:
"""
import numpy as np
import networkx as nx
import os, time

DATA_DIR = os.path.join(os.path.dirname(__file__), 'graph_data')

HARD_PAIRS = [
    (188999, 237547), (11370338, 11365103), (9031838, 8879174),
    (1814696, 2216909), (9185981, 9195054), (7029732, 6994332),
    (10867218, 10867304), (276924, 187619), (12004262, 12004272),
    (2216906, 2206763), (11828565, 11820028), (10708098, 10783003),
    (4078903, 7094156), (31466, 32109), (506838, 490842),
    (10827766, 10865091), (32110, 31467), (10920267, 10917913),
    (6894353, 6894749), (11993125, 11993028), (12001009, 12000911),
    (11830081, 11826877),
]


def parse_graph6(line):
    line = line.strip()
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


def load_specific(filepath, indices):
    """Load only specific graph indices from g6 file."""
    idx_set = set(indices)
    graphs = {}
    with open(filepath, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            if i in idx_set:
                graphs[i] = parse_graph6(line)
            if len(graphs) == len(idx_set):
                break
    return graphs


def graph_props(adj):
    n = len(adj)
    G = nx.from_numpy_array(adj.astype(int))
    e = int(adj.sum()) // 2
    degs = tuple(sorted(adj.sum(axis=1).astype(int).tolist()))
    eigs = sorted(np.linalg.eigvalsh(adj.astype(float)).tolist(), reverse=True)
    conn = nx.is_connected(G)
    # Check if self-complementary
    comp_adj = 1 - adj.astype(int) - np.eye(n, dtype=int)
    comp_G = nx.from_numpy_array(comp_adj)
    self_comp = nx.is_isomorphic(G, comp_G)
    # Check if complement-pair
    comp_of_G = comp_adj
    return {
        'edges': e, 'degs': degs, 'eigs': eigs, 'connected': conn,
        'self_comp': self_comp, 'adj': adj, 'comp_adj': comp_of_G.astype(np.uint8)
    }


if __name__ == '__main__':
    # Get all unique indices
    all_idx = sorted(set(i for pair in HARD_PAIRS for i in pair))
    print(f"Loading {len(all_idx)} specific graphs from n=10 catalog...")
    t0 = time.time()
    graphs = load_specific(os.path.join(DATA_DIR, 'graph10_decompressed.g6'), all_idx)
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # Build index-to-catalog-position for complement checking
    idx_to_idx = {i: i for i in all_idx}

    print(f"{'='*70}")
    print(f"  n=10 HARD PAIRS (k=6 fails, k=7 resolves)")
    print(f"{'='*70}")

    complement_pairs = 0
    srg_pairs = 0

    for i, (idx1, idx2) in enumerate(HARD_PAIRS):
        g1 = graphs[idx1]
        g2 = graphs[idx2]
        p1 = graph_props(g1)
        p2 = graph_props(g2)

        # Check if they're complement pairs
        G1 = nx.from_numpy_array(g1.astype(int))
        G2 = nx.from_numpy_array(g2.astype(int))
        comp1 = nx.from_numpy_array(p1['comp_adj'].astype(int))
        is_complement = nx.is_isomorphic(comp1, G2)

        assert not nx.is_isomorphic(G1, G2)

        marker = ""
        if is_complement:
            complement_pairs += 1
            marker = " [COMPLEMENT PAIR]"

        # Check if SRG (regular, and triangle+codegree regular)
        if p1['degs'][0] == p1['degs'][-1]:  # regular
            d = p1['degs'][0]
            # Check lambda (triangles per edge)
            tris = []
            for u in range(10):
                for v in range(10):
                    if g1[u, v]:
                        cn = sum(1 for w in range(10) if g1[u, w] and g1[v, w])
                        tris.append(cn)
            lam = tris[0] if len(set(tris)) == 1 else None
            # Check mu (paths of length 2 between non-adjacent)
            mus = []
            for u in range(10):
                for v in range(10):
                    if not g1[u, v] and u != v:
                        cn = sum(1 for w in range(10) if g1[u, w] and g1[v, w])
                        mus.append(cn)
            mu = mus[0] if len(set(mus)) == 1 else None
            if lam is not None and mu is not None:
                srg_pairs += 1
                marker += f" [SRG({10},{d},{lam},{mu})]"

        print(f"\nPair {i+1}: graphs {idx1} <-> {idx2}{marker}")
        e1, e2 = p1['edges'], p2['edges']
        print(f"  G{idx1}: {e1} edges, deg={p1['degs']}, conn={p1['connected']}")
        eig_str1 = [f"{x:.3f}" for x in p1['eigs']]
        print(f"    eigs: [{', '.join(eig_str1)}]")
        if p1['self_comp']:
            print(f"    SELF-COMPLEMENTARY!")
        print(f"  G{idx2}: {e2} edges, deg={p2['degs']}, conn={p2['connected']}")
        eig_str2 = [f"{x:.3f}" for x in p2['eigs']]
        print(f"    eigs: [{', '.join(eig_str2)}]")

        # Check same eigenvalues
        same_eigs = all(abs(p1['eigs'][j] - p2['eigs'][j]) < 0.01 for j in range(10))
        if same_eigs:
            print(f"  COSPECTRAL!")

    print(f"\n{'='*70}")
    print(f"Summary of 22 hard pairs:")
    print(f"  Complement pairs: {complement_pairs}")
    print(f"  SRG pairs: {srg_pairs}")
    print(f"  Cospectral pairs: (count in output above)")
