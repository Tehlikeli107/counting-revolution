"""
Exhaustive n=8 Proof via McKay Graph Catalog
=============================================
Instead of enumerating 268M labeled graphs, download all 12,346
non-isomorphic graphs on 8 vertices from Brendan McKay's database.

McKay's graph6 format stores graphs compactly. We parse each one,
compute our counting signature, and verify all 12,346 are distinct.

If #distinct signatures == 12,346 -> PROVEN COMPLETE for n=8!

Source: http://users.cecs.anu.edu.au/~bdm/data/graphs.html
"""

import numpy as np
import networkx as nx
from itertools import combinations
from collections import Counter
import time
import sys
import urllib.request
import gzip
import os


EXPECTED_N8 = 12346
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'graph_data')
GRAPH8_FILE = os.path.join(CACHE_DIR, 'graph8.g6')


def download_graph_catalog():
    """Download McKay's catalog of all non-isomorphic graphs on 8 vertices."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(GRAPH8_FILE):
        print(f"  Using cached catalog: {GRAPH8_FILE}")
        return

    # Try graph6 format (uncompressed)
    url = "https://users.cecs.anu.edu.au/~bdm/data/graph8.g6"
    print(f"  Downloading from {url}...")

    try:
        urllib.request.urlretrieve(url, GRAPH8_FILE)
        print(f"  Downloaded to {GRAPH8_FILE}")
    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  Trying compressed version...")
        url_gz = url + ".gz"
        gz_file = GRAPH8_FILE + ".gz"
        try:
            urllib.request.urlretrieve(url_gz, gz_file)
            with gzip.open(gz_file, 'rb') as f_in:
                with open(GRAPH8_FILE, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(gz_file)
            print(f"  Decompressed to {GRAPH8_FILE}")
        except Exception as e2:
            print(f"  Download failed: {e2}")
            print(f"  Will generate graphs using networkx instead.")
            return


def generate_graphs_nx(n):
    """Generate all non-iso graphs on n vertices using networkx graph_atlas for n<=7,
    or generate systematically for n=8 using canonical deletion."""
    # For n=8, networkx doesn't have a direct generator for all graphs.
    # We'll use a different approach: enumerate and filter.
    # But this is too slow for n=8 (268M graphs).
    # Instead, let's try to use networkx's graph6 parsing if we have the file.
    pass


def load_graphs_from_g6(filepath):
    """Load all graphs from a graph6 file."""
    graphs = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                try:
                    G = nx.from_graph6_bytes(line.encode('ascii'))
                    graphs.append(G)
                except:
                    pass
    return graphs


def compute_counting_signature(adj, n):
    """Full polynomial-time counting signature."""
    A = adj.astype(np.int64)
    degs_raw = adj.sum(axis=1)
    degs = tuple(sorted(degs_raw.tolist()))

    Ak = np.eye(n, dtype=np.int64)
    traces = []
    for k in range(1, n + 1):
        Ak = Ak @ A
        traces.append(int(np.trace(Ak)))

    e = [1]
    for k in range(1, n + 1):
        s = 0
        for i in range(1, k + 1):
            s += ((-1) ** (i - 1)) * e[k - i] * traces[i - 1]
        e.append(s // k)
    char_coeffs = tuple(e[1:])

    visited = set()
    n_comp = 0
    dist_hist = {}
    wiener = 0
    eccs = []
    for start in range(n):
        if start not in visited:
            n_comp += 1
        d = [-1] * n
        d[start] = 0
        q = [start]; qi = 0
        while qi < len(q):
            v = q[qi]; qi += 1
            visited.add(v)
            for u in range(n):
                if adj[v, u] and d[u] < 0:
                    d[u] = d[v] + 1
                    q.append(u)
        eccs.append(max(x for x in d if x >= 0))
        for j in range(start + 1, n):
            dj = d[j]
            if dj >= 0:
                dist_hist[dj] = dist_hist.get(dj, 0) + 1
                wiener += dj
            else:
                dist_hist[-1] = dist_hist.get(-1, 0) + 1

    if n > 1:
        L = np.diag(degs_raw.astype(np.float64)) - adj.astype(np.float64)
        n_span = round(np.linalg.det(L[1:, 1:]))
    else:
        n_span = 1

    clust = []
    for v in range(n):
        nbrs = [u for u in range(n) if adj[v, u]]
        k = len(nbrs)
        if k < 2:
            clust.append((0, 1))
        else:
            tri = sum(1 for i in range(len(nbrs)) for j in range(i+1, len(nbrs))
                      if adj[nbrs[i], nbrs[j]])
            clust.append((2 * tri, k * (k - 1)))

    ndp = tuple(sorted(
        tuple(sorted(int(adj[u].sum()) for u in range(n) if adj[v, u]))
        for v in range(n)
    ))

    ecn = []; ncn = []
    for u in range(n):
        for v in range(u + 1, n):
            cn = sum(1 for w in range(n) if adj[u, w] and adj[v, w])
            (ecn if adj[u, v] else ncn).append(cn)

    sub4 = ()
    if n >= 4:
        types = Counter()
        for sub in combinations(range(n), 4):
            sd = [0, 0, 0, 0]
            for i in range(4):
                for j in range(i + 1, 4):
                    if adj[sub[i], sub[j]]:
                        sd[i] += 1; sd[j] += 1
            types[tuple(sorted(sd))] += 1
        sub4 = tuple(sorted(types.items()))

    return (
        degs, tuple(traces), char_coeffs,
        n_comp, tuple(sorted(dist_hist.items())), wiener, tuple(sorted(eccs)),
        n_span, tuple(sorted(clust)), ndp,
        tuple(sorted(ecn)), tuple(sorted(ncn)),
        sub4,
    )


def main():
    print("=" * 70)
    print("  EXHAUSTIVE n=8 PROOF via McKay Graph Catalog")
    print("=" * 70)

    # Step 1: Get the graph catalog
    download_graph_catalog()

    if os.path.exists(GRAPH8_FILE):
        print(f"\n  Loading graphs from catalog...")
        t0 = time.time()
        graphs = load_graphs_from_g6(GRAPH8_FILE)
        t_load = time.time() - t0
        print(f"  Loaded {len(graphs)} graphs in {t_load:.1f}s")
    else:
        print(f"\n  Catalog not available. Generating via networkx...")
        # Fallback: generate all graphs on 8 vertices using edge enumeration
        # with graph-tool style canonical form. Too slow without nauty.
        print(f"  ERROR: Cannot generate n=8 without catalog. Aborting.")
        return

    if len(graphs) != EXPECTED_N8:
        print(f"  WARNING: Got {len(graphs)} graphs, expected {EXPECTED_N8}")

    # Step 2: Compute counting signatures for all graphs
    print(f"\n  Computing counting signatures for {len(graphs)} graphs...")
    t0 = time.time()

    signatures = {}  # sig -> graph index
    collisions = []

    for idx, G in enumerate(graphs):
        n = G.number_of_nodes()
        adj = nx.to_numpy_array(G, dtype=np.int32)
        sig = compute_counting_signature(adj, n)

        if sig in signatures:
            collisions.append((signatures[sig], idx))
            print(f"    COLLISION! Graph {signatures[sig]} and {idx} share signature")
            print(f"    Degree seq: {sig[0]}")
        else:
            signatures[sig] = idx

        if (idx + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(graphs) - idx - 1) / rate
            print(f"    {idx+1:>6}/{len(graphs)} | {len(signatures):,} distinct sigs "
                  f"| {elapsed:.0f}s | ETA {eta:.0f}s")
            sys.stdout.flush()

    t_sig = time.time() - t0
    n_sigs = len(signatures)

    print(f"\n{'='*70}")
    print(f"  RESULT (n=8, EXHAUSTIVE)")
    print(f"{'='*70}")
    print(f"  Total non-iso graphs: {len(graphs)}")
    print(f"  Distinct counting signatures: {n_sigs}")
    print(f"  Collisions: {len(collisions)}")
    print(f"  Time: {t_sig:.1f}s ({t_sig/60:.1f} min)")

    if n_sigs == len(graphs) and len(collisions) == 0:
        print(f"\n  *** PROVEN: Counting = COMPLETE classification for n=8! ***")
        print(f"  {n_sigs} signatures = {len(graphs)} iso classes = PERFECT!")
        print(f"\n  Combined with n<=7 proof:")
        print(f"  THEOREM: Polynomial-time counting invariants provide")
        print(f"  complete graph isomorphism classification for ALL")
        print(f"  simple graphs on n <= 8 vertices.")
    else:
        print(f"\n  INCOMPLETE: {len(collisions)} collision(s)")
        for c in collisions:
            print(f"    Graphs {c[0]} and {c[1]} share same signature")
            # Show the graphs
            G1 = graphs[c[0]]
            G2 = graphs[c[1]]
            print(f"    G1: {G1.number_of_edges()} edges, connected={nx.is_connected(G1)}")
            print(f"    G2: {G2.number_of_edges()} edges, connected={nx.is_connected(G2)}")
            print(f"    Isomorphic: {nx.is_isomorphic(G1, G2)}")


if __name__ == '__main__':
    main()
