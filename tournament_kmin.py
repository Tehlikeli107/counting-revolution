"""
Tournament Counting Revolution
==============================
Question: What minimum k of induced k-sub-tournament counts classifies
all non-isomorphic tournaments on n vertices?

Tournament: complete directed graph (for each pair, exactly one direction).
n vertices -> 2^C(n,2) tournaments -> many fewer non-iso classes.

Non-iso tournament counts: 1,1,2,4,12,56,456,6880,... (OEIS A000568)
  n=1:1, n=2:1, n=3:2, n=4:4, n=5:12, n=6:56, n=7:456, n=8:6880

Key question: k_min for tournaments vs k_min for undirected graphs.
Hypothesis: tournaments might have smaller k_min since direction = extra info.
"""
import numpy as np
from itertools import combinations, permutations
from collections import defaultdict
import time


def build_tournament_lookup(k):
    """Canonical type lookup for directed k-vertex tournaments.

    Represent each tournament on k vertices by a bitmask of k*(k-1)/2 bits:
    for each pair (i,j) with i<j, bit=0 means i->j, bit=1 means j->i.
    Canonical = minimum bitmask over all k! vertex relabelings.
    """
    pairs = [(i,j) for i in range(k) for j in range(i+1,k)]
    n_pairs = len(pairs)
    pair_idx = {p: i for i,p in enumerate(pairs)}

    # Precompute permutation mappings: for each perm, src_bit -> dst_bit + flip?
    # When applying perm to tournament:
    #   original pair (i,j) -> new pair (perm[i], perm[j])
    #   normalize: if perm[i]<perm[j], direction preserved; else flipped

    N_pats = 1 << n_pairs
    canon_to_type = {}; type_id = 0
    arr = np.zeros(N_pats, dtype=np.int32)

    for bits in range(N_pats):
        min_bits = bits
        for perm in permutations(range(k)):
            new_bits = 0
            for idx,(i,j) in enumerate(pairs):
                pi, pj = perm[i], perm[j]
                if pi < pj:
                    # pair (pi,pj) is canonical; check if original says j->i
                    dst = pair_idx[(pi,pj)]
                    # Original: bit idx=0 means i->j, bit=1 means j->i
                    # Under perm: i->j becomes pi->pj
                    # If (bits>>idx)&1: j->i -> pj->pi -> bit=1 in (pj,pi) pair
                    # But (pi,pj) is canonical, so we need pi->pj direction
                    # (pi,pj): bit=0 means pi->pj
                    # Was j->i? if yes, perm gives pj->pi, so (pi,pj) bit = 1
                    if (bits >> idx) & 1:
                        new_bits |= (1 << dst)
                    # else: i->j -> pi->pj, (pi,pj) bit = 0 (already 0)
                else:
                    # pair (pj,pi) is canonical; original direction flipped
                    dst = pair_idx[(pj,pi)]
                    # Original: i->j if bit=0, j->i if bit=1
                    # Under perm: i->j -> pi->pj = pj->pi reversed -> (pj,pi) bit=1
                    # Under perm: j->i -> pj->pi -> (pj,pi) bit=0
                    if not ((bits >> idx) & 1):
                        new_bits |= (1 << dst)
            min_bits = min(min_bits, new_bits)

        if min_bits not in canon_to_type:
            canon_to_type[min_bits] = type_id; type_id += 1
        arr[bits] = canon_to_type[min_bits]

    return arr, type_id, pairs


def canonical_tournament(A, n):
    """Canonical form of tournament A (adjacency matrix) on n vertices."""
    pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    pair_idx = {p: i for i,p in enumerate(pairs)}
    n_pairs = len(pairs)

    bits0 = 0
    for idx,(i,j) in enumerate(pairs):
        if A[j,i]:  # j->i
            bits0 |= (1 << idx)

    min_bits = bits0
    for perm in permutations(range(n)):
        new_bits = 0
        for idx,(i,j) in enumerate(pairs):
            pi,pj = perm[i],perm[j]
            if pi < pj:
                dst = pair_idx[(pi,pj)]
                if A[j,i]:  # j->i -> pj->pi... but pi<pj so (pi,pj) bit=1
                    new_bits |= (1 << dst)
            else:
                dst = pair_idx[(pj,pi)]
                if A[i,j]:  # i->j -> pi->pj -> pj<pi reversed -> (pj,pi) bit=0, else bit=1
                    # Actually: i->j under perm gives pi->pj. Since pj<pi, canonical is (pj,pi).
                    # Direction pi->pj = direction NOT from pj to pi, so (pj,pi) bit=1.
                    new_bits |= (1 << dst)
        min_bits = min(min_bits, new_bits)
    return min_bits


def enumerate_non_iso_tournaments(n):
    """Enumerate all non-isomorphic tournaments on n vertices."""
    pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    n_pairs = len(pairs)
    seen = set()
    tourns = []
    for bits in range(1 << n_pairs):
        A = np.zeros((n,n), dtype=np.int8)
        for idx,(i,j) in enumerate(pairs):
            if (bits >> idx) & 1:
                A[j,i] = 1
            else:
                A[i,j] = 1
        canon = canonical_tournament(A, n)
        if canon not in seen:
            seen.add(canon)
            tourns.append(A)
    return tourns


def compute_tournament_sig(A, n, k, lookup, num_types, pairs_k):
    counts = np.zeros(num_types, dtype=np.int32)
    n_pairs = k*(k-1)//2
    pair_idx_k = {p: i for i,p in enumerate(pairs_k)}
    for sub in combinations(range(n), k):
        bits = 0
        for idx,(i,j) in enumerate(pairs_k):
            vi,vj = sub[i],sub[j]
            if A[vj,vi]:  # j->i in subgraph context -> vj->vi in original
                bits |= (1 << idx)
        counts[lookup[bits]] += 1
    return tuple(counts)


if __name__ == '__main__':
    print("Tournament Counting Revolution")
    print("=" * 60)
    print("Non-iso counts: n=1:1, n=2:1, n=3:2, n=4:4, n=5:12, n=6:56")
    print()

    results = {}
    for n in range(3, 8):
        t0 = time.time()
        tourns = enumerate_non_iso_tournaments(n)
        enum_t = time.time() - t0
        print(f"n={n}: {len(tourns)} non-iso tournaments (enum: {enum_t:.1f}s)")

        # Test k_min
        found_k = None
        for k in range(2, n):
            t0 = time.time()
            lookup, num_types, pairs_k = build_tournament_lookup(k)
            build_t = time.time() - t0
            print(f"  k={k}: {num_types} tournament types ({build_t:.1f}s)", end='', flush=True)

            t0 = time.time()
            sigs = [compute_tournament_sig(A, n, k, lookup, num_types, pairs_k)
                    for A in tourns]
            sig_t = time.time() - t0

            n_distinct = len(set(sigs))
            collisions = len(sigs) - n_distinct
            print(f" -> {n_distinct}/{len(sigs)} distinct ({collisions} collisions, {sig_t:.1f}s)")

            if collisions == 0:
                found_k = k
                break

        results[n] = (len(tourns), found_k)
        print(f"  => k_min(tournaments, n={n}) <= {found_k}")
        print()

    print("Summary:")
    print(f"{'n':>4} {'#tournaments':>14} {'k_min':>7} {'relation':>10}")
    for n, (cnt, k) in results.items():
        rel = f"n-{n-k}" if k else "?"
        print(f"{n:>4} {cnt:>14} {k if k else '?':>7} {rel:>10}")
