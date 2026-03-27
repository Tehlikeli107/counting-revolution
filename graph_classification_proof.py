"""
PROOF: Counting Invariants = Complete Graph Classification for n<=7
=====================================================================

Approach (no canonical form needed!):
1. Enumerate ALL 2^(n choose 2) labeled graphs
2. Compute counting signature for each
3. Group by signature
4. Within each group, check isomorphism (using nx.is_isomorphic)
5. If every group = single iso class -> COUNTING IS COMPLETE

For n=7: 2^21 = 2,097,152 labeled graphs, 1,044 iso classes.
The signature computation is fast (~O(n^4) per graph with 4-subgraph counting).
"""

import numpy as np
import networkx as nx
from itertools import combinations
from collections import Counter, defaultdict
import time
import sys


EXPECTED = {4: 11, 5: 34, 6: 156, 7: 1044, 8: 12346}


def compute_counting_signature(adj, n):
    """Polynomial-time counting signature. O(n^4) with 4-subgraph counting."""
    A = adj.astype(np.int64)
    degs_raw = adj.sum(axis=1)
    degs = tuple(sorted(degs_raw.tolist()))

    # Traces A^1..A^n
    Ak = np.eye(n, dtype=np.int64)
    traces = []
    for k in range(1, n + 1):
        Ak = Ak @ A
        traces.append(int(np.trace(Ak)))

    # Char poly via Newton's identities
    e = [1]
    for k in range(1, n + 1):
        s = 0
        for i in range(1, k + 1):
            s += ((-1) ** (i - 1)) * e[k - i] * traces[i - 1]
        e.append(s // k)
    char_coeffs = tuple(e[1:])

    # BFS: components, distance hist, wiener, eccentricity
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

    # Spanning trees
    if n > 1:
        L = np.diag(degs_raw.astype(np.float64)) - adj.astype(np.float64)
        n_span = round(np.linalg.det(L[1:, 1:]))
    else:
        n_span = 1

    # Clustering (rational)
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

    # Neighbor degree profile
    ndp = tuple(sorted(
        tuple(sorted(int(adj[u].sum()) for u in range(n) if adj[v, u]))
        for v in range(n)
    ))

    # Edge/non-edge common neighbor profiles
    ecn = []; ncn = []
    for u in range(n):
        for v in range(u + 1, n):
            cn = sum(1 for w in range(n) if adj[u, w] and adj[v, w])
            (ecn if adj[u, v] else ncn).append(cn)

    # 4-vertex induced subgraph profile
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
    print("  PROOF: Counting = Complete Graph Classification")
    print("  Method: Group by signature, verify iso within groups")
    print("=" * 70)

    for n in [7]:
        print(f"\n{'='*70}")
        print(f"  n = {n}")
        print(f"{'='*70}")

        n_edges = n * (n - 1) // 2
        total = 2 ** n_edges
        expected = EXPECTED[n]
        edge_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        print(f"  Total labeled graphs: {total:,}")
        print(f"  Expected iso classes: {expected}")
        print(f"  Computing counting signatures for all graphs...")

        # Step 1: compute signature for every labeled graph
        sig_groups = defaultdict(list)  # sig -> list of adj matrices
        t0 = time.time()

        for mask in range(total):
            adj = np.zeros((n, n), dtype=np.int32)
            for bit, (i, j) in enumerate(edge_pairs):
                if mask & (1 << bit):
                    adj[i, j] = adj[j, i] = 1

            sig = compute_counting_signature(adj, n)
            sig_groups[sig].append(adj)

            if (mask + 1) % 200000 == 0:
                elapsed = time.time() - t0
                rate = (mask + 1) / elapsed
                eta = (total - mask - 1) / rate
                print(f"    {mask+1:>10,}/{total:,} | {len(sig_groups):,} sigs "
                      f"| {elapsed:.0f}s | ETA {eta:.0f}s")
                sys.stdout.flush()

        t_sig = time.time() - t0
        n_sigs = len(sig_groups)
        print(f"\n  Signatures computed in {t_sig:.0f}s")
        print(f"  Distinct signatures: {n_sigs}")

        # Step 2: verify iso within each group
        print(f"\n  Verifying isomorphism within each signature group...")
        t0 = time.time()

        n_iso_classes = 0
        max_group_size = 0
        failed_groups = 0

        for sig_idx, (sig, adjs) in enumerate(sig_groups.items()):
            # Convert to networkx graphs
            graphs = [nx.from_numpy_array(a) for a in adjs]

            # Find iso classes within this group
            classes = []
            for g in graphs:
                found = False
                for cls in classes:
                    if nx.is_isomorphic(g, cls[0]):
                        cls.append(g)
                        found = True
                        break
                if not found:
                    classes.append([g])

            n_iso_classes += len(classes)
            max_group_size = max(max_group_size, len(adjs))

            if len(classes) > 1:
                failed_groups += 1
                sizes = [len(c) for c in classes]
                print(f"    SPLIT in group {sig_idx}: {len(adjs)} graphs -> "
                      f"{len(classes)} iso classes (sizes: {sizes})")

            if (sig_idx + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"    Verified {sig_idx+1}/{n_sigs} groups | "
                      f"{n_iso_classes} iso classes | {elapsed:.0f}s")
                sys.stdout.flush()

        t_verify = time.time() - t0
        print(f"\n  Verification complete in {t_verify:.0f}s")
        print(f"  Max group size: {max_group_size}")

        print(f"\n{'='*70}")
        print(f"  RESULT for n={n}:")
        print(f"  Distinct counting signatures: {n_sigs}")
        print(f"  Distinct iso classes (verified): {n_iso_classes}")
        print(f"  Expected iso classes: {expected}")
        print(f"  Failed groups (multi-class): {failed_groups}")

        if n_sigs == n_iso_classes == expected:
            print(f"\n  *** PROVEN: Counting = COMPLETE classification for n={n}! ***")
            print(f"  {n_sigs} signatures = {expected} iso classes = PERFECT!")
        elif n_sigs == n_iso_classes:
            print(f"\n  Counting = complete ({n_sigs} = {n_iso_classes})")
            print(f"  But count differs from expected ({expected})!")
        else:
            print(f"\n  Counting INCOMPLETE: {n_sigs} sigs < {n_iso_classes} iso classes")
            print(f"  {failed_groups} signature groups contain multiple iso classes")


if __name__ == '__main__':
    main()
