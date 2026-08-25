"""
Finite n=8 validation via Brendan McKay's complete graph catalog.

This script evaluates the repository's historical 13-component counting signature
on all 12,346 non-isomorphic simple graphs on 8 vertices in McKay's catalog.

The validated publication evidence is committed separately at:

- benchmarks/publication-evidence/2026-08-25-v51/
- benchmarks/publication-evidence/2026-08-25-v54/

Later finite-order boundary/refinement evidence is at:

- benchmarks/publication-evidence/2026-08-25-v59/
- benchmarks/publication-evidence/2026-08-25-v64/
- benchmarks/publication-evidence/2026-08-25-v67/

Scope: a collision-free result here is a finite statement about this exact
signature on the complete order-8 catalog. It is not a general graph-isomorphism
theorem, an asymptotic result, or a novelty/priority claim.
"""

from collections import Counter
from itertools import combinations
import hashlib
import os
import sys
import time
import urllib.request

import networkx as nx
import numpy as np


EXPECTED_N8 = 12_346
GRAPH8_URL = "https://users.cecs.anu.edu.au/~bdm/data/graph8.g6"
EXPECTED_GRAPH8_BYTES = 86_422
EXPECTED_GRAPH8_SHA256 = (
    "546a249902101c97d3aa590f93e53366854bd0a6f405aa59bdb32d25c57f845a"
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "graph_data")
GRAPH8_FILE = os.path.join(CACHE_DIR, "graph8.g6")


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_graph_catalog(filepath):
    """Require the exact graph8.g6 bytes used by the committed evidence."""
    size = os.path.getsize(filepath)
    digest = sha256_file(filepath)

    if size != EXPECTED_GRAPH8_BYTES:
        raise RuntimeError(
            f"graph8.g6 byte-count mismatch: got {size}, "
            f"expected {EXPECTED_GRAPH8_BYTES}"
        )
    if digest != EXPECTED_GRAPH8_SHA256:
        raise RuntimeError(
            "graph8.g6 SHA-256 mismatch: "
            f"got {digest}, expected {EXPECTED_GRAPH8_SHA256}"
        )


def download_graph_catalog():
    """Return a locally cached, checksum-validated McKay order-8 catalog."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(GRAPH8_FILE):
        validate_graph_catalog(GRAPH8_FILE)
        print(f"  Using checksum-validated catalog: {GRAPH8_FILE}")
        return GRAPH8_FILE

    temp_file = GRAPH8_FILE + ".download"
    if os.path.exists(temp_file):
        os.remove(temp_file)

    print(f"  Downloading exact catalog from {GRAPH8_URL}...")
    try:
        urllib.request.urlretrieve(GRAPH8_URL, temp_file)
        validate_graph_catalog(temp_file)
        os.replace(temp_file, GRAPH8_FILE)
    except Exception:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

    print(f"  Downloaded and validated: {GRAPH8_FILE}")
    return GRAPH8_FILE


def load_graphs_from_g6(filepath):
    """Parse every graph6 record; malformed records are fatal, not skipped."""
    validate_graph_catalog(filepath)

    graphs = []
    with open(filepath, "rb") as f:
        for line_no, raw in enumerate(f, 1):
            record = raw.rstrip(b"\r\n")
            if not record:
                raise RuntimeError(f"blank graph6 record at line {line_no}")
            if record.startswith(b">>graph6<<"):
                raise RuntimeError(
                    f"unexpected graph6 header at line {line_no}; "
                    "the provenance-bound graph8.g6 capture has no header"
                )
            try:
                graph = nx.from_graph6_bytes(record)
            except Exception as exc:
                raise RuntimeError(
                    f"invalid graph6 record at line {line_no}: {exc}"
                ) from exc

            if graph.number_of_nodes() != 8:
                raise RuntimeError(
                    f"unexpected graph order at line {line_no}: "
                    f"{graph.number_of_nodes()}"
                )
            graphs.append(graph)

    if len(graphs) != EXPECTED_N8:
        raise RuntimeError(
            f"catalog record-count mismatch: got {len(graphs)}, expected {EXPECTED_N8}"
        )

    return graphs


def bareiss_det(matrix):
    """Exact integer determinant using fraction-free Bareiss elimination."""
    a = [list(map(int, row)) for row in matrix]
    n = len(a)

    if n == 0:
        return 1
    if n == 1:
        return a[0][0]

    sign = 1
    previous_pivot = 1

    for k in range(n - 1):
        if a[k][k] == 0:
            swap = next(
                (row for row in range(k + 1, n) if a[row][k] != 0),
                None,
            )
            if swap is None:
                return 0
            a[k], a[swap] = a[swap], a[k]
            sign *= -1

        pivot = a[k][k]

        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = a[i][j] * pivot - a[i][k] * a[k][j]
                if k > 0:
                    if numerator % previous_pivot != 0:
                        raise ArithmeticError(
                            "Bareiss exact division failed unexpectedly"
                        )
                    numerator //= previous_pivot
                a[i][j] = numerator

        previous_pivot = pivot

        for i in range(k + 1, n):
            a[i][k] = 0
        for j in range(k + 1, n):
            a[k][j] = 0

    return sign * a[n - 1][n - 1]


def exact_spanning_tree_count(adj):
    """Exact Matrix-Tree theorem count."""
    n = adj.shape[0]
    if n <= 1:
        return 1

    a = adj.astype(np.int64, copy=False)
    degrees = a.sum(axis=1).astype(np.int64)
    laplacian = np.diag(degrees) - a
    return int(bareiss_det(laplacian[1:, 1:].tolist()))


def compute_counting_signature(adj, n):
    """Compute the historical 13 top-level counting-signature components."""
    a = adj.astype(np.int64)
    degs_raw = adj.sum(axis=1)
    degs = tuple(sorted(degs_raw.tolist()))

    ak = np.eye(n, dtype=np.int64)
    traces = []
    for _ in range(1, n + 1):
        ak = ak @ a
        traces.append(int(np.trace(ak)))

    elementary = [1]
    for k in range(1, n + 1):
        total = 0
        for i in range(1, k + 1):
            total += (
                ((-1) ** (i - 1))
                * elementary[k - i]
                * traces[i - 1]
            )
        if total % k != 0:
            raise ArithmeticError(
                f"Newton-identity exact division failed at coefficient {k}"
            )
        elementary.append(total // k)
    char_coeffs = tuple(elementary[1:])

    visited = set()
    n_comp = 0
    dist_hist = {}
    wiener = 0
    eccs = []

    for start in range(n):
        if start not in visited:
            n_comp += 1

        distance = [-1] * n
        distance[start] = 0
        queue = [start]
        qi = 0

        while qi < len(queue):
            vertex = queue[qi]
            qi += 1
            visited.add(vertex)

            for neighbor in range(n):
                if adj[vertex, neighbor] and distance[neighbor] < 0:
                    distance[neighbor] = distance[vertex] + 1
                    queue.append(neighbor)

        eccs.append(max(x for x in distance if x >= 0))

        for j in range(start + 1, n):
            d = distance[j]
            if d >= 0:
                dist_hist[d] = dist_hist.get(d, 0) + 1
                wiener += d
            else:
                dist_hist[-1] = dist_hist.get(-1, 0) + 1

    n_span = exact_spanning_tree_count(adj)

    clustering = []
    for vertex in range(n):
        neighbors = [u for u in range(n) if adj[vertex, u]]
        degree = len(neighbors)

        if degree < 2:
            clustering.append((0, 1))
        else:
            triangles = sum(
                1
                for i in range(len(neighbors))
                for j in range(i + 1, len(neighbors))
                if adj[neighbors[i], neighbors[j]]
            )
            clustering.append((2 * triangles, degree * (degree - 1)))

    neighbor_degree_profile = tuple(
        sorted(
            tuple(
                sorted(
                    int(adj[u].sum())
                    for u in range(n)
                    if adj[vertex, u]
                )
            )
            for vertex in range(n)
        )
    )

    edge_common_neighbors = []
    nonedge_common_neighbors = []

    for u in range(n):
        for v in range(u + 1, n):
            common = sum(
                1
                for w in range(n)
                if adj[u, w] and adj[v, w]
            )
            if adj[u, v]:
                edge_common_neighbors.append(common)
            else:
                nonedge_common_neighbors.append(common)

    induced4_profile = ()
    if n >= 4:
        types = Counter()

        for subset in combinations(range(n), 4):
            sub_degrees = [0, 0, 0, 0]

            for i in range(4):
                for j in range(i + 1, 4):
                    if adj[subset[i], subset[j]]:
                        sub_degrees[i] += 1
                        sub_degrees[j] += 1

            types[tuple(sorted(sub_degrees))] += 1

        induced4_profile = tuple(sorted(types.items()))

    return (
        degs,
        tuple(traces),
        char_coeffs,
        n_comp,
        tuple(sorted(dist_hist.items())),
        wiener,
        tuple(sorted(eccs)),
        n_span,
        tuple(sorted(clustering)),
        neighbor_degree_profile,
        tuple(sorted(edge_common_neighbors)),
        tuple(sorted(nonedge_common_neighbors)),
        induced4_profile,
    )


def main():
    print("=" * 76)
    print("  FINITE n=8 VALIDATION via McKay complete graph catalog")
    print("=" * 76)

    catalog_path = download_graph_catalog()

    print("\n  Loading checksum-bound catalog...")
    load_start = time.time()
    graphs = load_graphs_from_g6(catalog_path)
    load_elapsed = time.time() - load_start
    print(f"  Loaded {len(graphs):,} graphs in {load_elapsed:.1f}s")

    print(f"\n  Computing 13-component signatures for {len(graphs):,} graphs...")
    signature_start = time.time()

    signatures = {}
    collisions = []

    for index, graph in enumerate(graphs):
        adjacency = nx.to_numpy_array(
            graph,
            nodelist=list(range(8)),
            dtype=np.int32,
        )
        signature = compute_counting_signature(adjacency, 8)

        if signature in signatures:
            collisions.append((signatures[signature], index))
            print(
                f"    COLLISION: catalog indices "
                f"{signatures[signature]} and {index}"
            )
        else:
            signatures[signature] = index

        if (index + 1) % 2000 == 0:
            elapsed = time.time() - signature_start
            rate = (index + 1) / elapsed
            remaining = (len(graphs) - index - 1) / rate
            print(
                f"    {index + 1:>6}/{len(graphs)} | "
                f"{len(signatures):,} distinct | "
                f"{elapsed:.0f}s | ETA {remaining:.0f}s"
            )
            sys.stdout.flush()

    signature_elapsed = time.time() - signature_start

    print(f"\n{'=' * 76}")
    print("  RESULT — finite complete order-8 catalog")
    print(f"{'=' * 76}")
    print(f"  Catalog graphs: {len(graphs):,}")
    print(f"  Distinct signatures: {len(signatures):,}")
    print(f"  Collision groups observed by this pass: {len(collisions):,}")
    print(
        f"  Time: {signature_elapsed:.1f}s "
        f"({signature_elapsed / 60:.1f} min)"
    )

    if len(signatures) == EXPECTED_N8 and not collisions:
        print(
            "\n  VALIDATED FINITE RESULT: this exact 13-component signature is "
            "collision-free on McKay's complete order-8 catalog."
        )
        print(
            "  Publication evidence: "
            "benchmarks/publication-evidence/2026-08-25-v51/"
        )
        print(
            "  Component-minimality evidence: "
            "benchmarks/publication-evidence/2026-08-25-v54/"
        )
        print(
            "\n  Scope: order 8 only. This output is not a general graph-"
            "isomorphism theorem and makes no claim for larger graph orders."
        )
    else:
        print(
            f"\n  INCOMPLETE ON THIS CATALOG: "
            f"{len(collisions):,} repeated signature(s)"
        )

        for first, second in collisions:
            graph1 = graphs[first]
            graph2 = graphs[second]
            print(f"    Catalog indices {first} and {second}")
            print(
                f"      G1: {graph1.number_of_edges()} edges, "
                f"connected={nx.is_connected(graph1)}"
            )
            print(
                f"      G2: {graph2.number_of_edges()} edges, "
                f"connected={nx.is_connected(graph2)}"
            )
            print(
                f"      NetworkX isomorphic: "
                f"{nx.is_isomorphic(graph1, graph2)}"
            )


if __name__ == "__main__":
    main()
