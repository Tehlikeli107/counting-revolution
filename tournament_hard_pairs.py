"""
Analyze the hard pairs in tournament classification.
WHY does k_min(tournament, n) = n?
- Reversal is confirmed for n=4, but not complete explanation for n>=5
- Find the structural property of hard pairs in n=5,6
"""
import numpy as np
from itertools import combinations, permutations
from collections import defaultdict


def build_tournament_lookup(k):
    pairs = [(i,j) for i in range(k) for j in range(i+1,k)]
    n_pairs = len(pairs)
    pair_idx = {p: i for i,p in enumerate(pairs)}
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
                    dst = pair_idx[(pi,pj)]
                    if (bits >> idx) & 1:
                        new_bits |= (1 << dst)
                else:
                    dst = pair_idx[(pj,pi)]
                    if not ((bits >> idx) & 1):
                        new_bits |= (1 << dst)
            min_bits = min(min_bits, new_bits)
        if min_bits not in canon_to_type:
            canon_to_type[min_bits] = type_id; type_id += 1
        arr[bits] = canon_to_type[min_bits]
    return arr, type_id, pairs


def canonical_tournament(A, n):
    pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    pair_idx = {p: i for i,p in enumerate(pairs)}
    bits0 = 0
    for idx,(i,j) in enumerate(pairs):
        if A[j,i]:
            bits0 |= (1 << idx)
    min_bits = bits0
    for perm in permutations(range(n)):
        new_bits = 0
        for idx,(i,j) in enumerate(pairs):
            pi,pj = perm[i],perm[j]
            if pi < pj:
                dst = pair_idx[(pi,pj)]
                if A[j,i]:
                    new_bits |= (1 << dst)
            else:
                dst = pair_idx[(pj,pi)]
                if A[i,j]:
                    new_bits |= (1 << dst)
        min_bits = min(min_bits, new_bits)
    return min_bits


def enumerate_non_iso_tournaments(n):
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
            tourns.append((A, bits))
    return tourns


def compute_sig(A, n, k, lookup, num_types, pairs_k):
    counts = np.zeros(num_types, dtype=np.int32)
    pair_idx_k = {p: i for i,p in enumerate(pairs_k)}
    for sub in combinations(range(n), k):
        bits = 0
        for idx,(i,j) in enumerate(pairs_k):
            vi,vj = sub[i],sub[j]
            if A[vj,vi]:
                bits |= (1 << idx)
        counts[lookup[bits]] += 1
    return tuple(counts)


def reverse_tournament(A, n):
    """Reverse all arcs: i->j becomes j->i."""
    return A.T.copy()


def score_sequence(A, n):
    """Out-degree sequence (sorted)."""
    return tuple(sorted(A.sum(axis=1)))


def is_isomorphic(A, B, n):
    """Check if A and B are isomorphic tournaments."""
    ca = canonical_tournament(A, n)
    cb = canonical_tournament(B, n)
    return ca == cb


def describe_tournament(A, n, idx):
    scores = score_sequence(A, n)
    out_deg = list(A.sum(axis=1))
    rev = reverse_tournament(A, n)
    is_self_dual = is_isomorphic(A, rev, n)
    return f"T{idx}: scores={scores}, self-dual={is_self_dual}"


def find_structural_symmetry(A, B, n):
    """
    Investigate if A and B are related by some algebraic symmetry
    beyond reversal/isomorphism.
    """
    results = {}

    # 1. Are they reversal pairs?
    rev_A = reverse_tournament(A, n)
    results['rev_A iso B'] = is_isomorphic(rev_A, B, n)

    # 2. Do they have same score sequence?
    results['same scores'] = score_sequence(A, n) == score_sequence(B, n)

    # 3. Closed walk counts (traces of powers of adjacency matrix)
    for p in range(2, n+1):
        try:
            Ap = np.linalg.matrix_power(A.astype(float), p)
            Bp = np.linalg.matrix_power(B.astype(float), p)
            results[f'tr(A^{p})==tr(B^{p})'] = abs(np.trace(Ap) - np.trace(Bp)) < 1e-6
        except:
            pass

    # 4. Eigenvalue spectrum of adjacency matrix
    eigA = sorted(np.linalg.eigvals(A.astype(float)).real.round(6))
    eigB = sorted(np.linalg.eigvals(B.astype(float)).real.round(6))
    results['same eigenvalues'] = eigA == eigB

    # 5. Score sequence of reverse tournaments
    results['same rev scores'] = score_sequence(rev_A, n) == score_sequence(reverse_tournament(B, n), n)

    # 6. Subtournament score sequences: for each k, check score seq distributions
    for k in range(3, n):
        score_dist_A = defaultdict(int)
        score_dist_B = defaultdict(int)
        for sub in combinations(range(n), k):
            sub = list(sub)
            sub_A = A[np.ix_(sub,sub)]
            sub_B = B[np.ix_(sub,sub)]
            sa = score_sequence(sub_A, k)
            sb = score_sequence(sub_B, k)
            score_dist_A[sa] += 1
            score_dist_B[sb] += 1
        results[f'same k={k} score dist'] = dict(score_dist_A) == dict(score_dist_B)

    return results


def analyze_hard_pairs(n):
    print(f"\n{'='*60}")
    print(f"Analyzing hard pairs for n={n} tournaments")
    print(f"{'='*60}")

    # Build lookups for all k < n
    lookups = {}
    for k in range(2, n):
        lookup, num_types, pairs_k = build_tournament_lookup(k)
        lookups[k] = (lookup, num_types, pairs_k)

    tourns = enumerate_non_iso_tournaments(n)
    print(f"Total non-iso tournaments: {len(tourns)}")

    # Find hard pairs: same sig for all k < n
    # Compute combined signatures
    all_sigs = []
    for A, bits in tourns:
        combined = []
        for k in range(2, n):
            lookup, num_types, pairs_k = lookups[k]
            sig = compute_sig(A, n, k, lookup, num_types, pairs_k)
            combined.extend(sig)
        all_sigs.append(tuple(combined))

    # Find collisions
    sig_to_idx = defaultdict(list)
    for i, sig in enumerate(all_sigs):
        sig_to_idx[sig].append(i)

    hard_groups = [(sig, idxs) for sig, idxs in sig_to_idx.items() if len(idxs) > 1]
    print(f"Hard collision groups (all k<{n} fail together): {len(hard_groups)}")

    for gi, (sig, idxs) in enumerate(hard_groups):
        print(f"\nHard group {gi+1}: {len(idxs)} tournaments (indices {idxs})")
        for i in idxs:
            A, bits = tourns[i]
            print(f"  {describe_tournament(A, n, i)}")

        # Analyze pair structure (take first pair)
        i, j = idxs[0], idxs[1]
        A, _ = tourns[i]
        B, _ = tourns[j]

        print(f"\n  Structural analysis of T{i} vs T{j}:")
        sym = find_structural_symmetry(A, B, n)
        for k, v in sym.items():
            print(f"    {k}: {v}")

        # Show adjacency matrices
        print(f"\n  T{i} adjacency (row i -> col j):")
        print(f"  {A.tolist()}")
        print(f"  T{j} adjacency:")
        print(f"  {B.tolist()}")

        # Show score sequences explicitly
        print(f"\n  T{i} out-degrees: {list(A.sum(axis=1))}")
        print(f"  T{j} out-degrees: {list(B.sum(axis=1))}")


def full_reversal_analysis(n):
    """For all non-iso tournaments, check which ones survive reversal."""
    print(f"\n{'='*60}")
    print(f"Reversal analysis for n={n}")
    print(f"{'='*60}")

    tourns = enumerate_non_iso_tournaments(n)

    # Build sig for k = n-1
    k = n-1
    lookup, num_types, pairs_k = build_tournament_lookup(k)

    sigs = {}
    for i, (A, bits) in enumerate(tourns):
        sig = compute_sig(A, n, k, lookup, num_types, pairs_k)
        sigs[i] = sig

    # Check reversal pairs
    self_dual = []
    reversal_pairs = []
    checked = set()

    for i, (A, bits) in enumerate(tourns):
        if i in checked:
            continue
        rev = reverse_tournament(A, n)
        rev_canon = canonical_tournament(rev, n)
        # Find which tournament index has this canonical form
        for j, (B, _) in enumerate(tourns):
            if j != i and canonical_tournament(B, n) == rev_canon:
                reversal_pairs.append((i, j))
                checked.add(i)
                checked.add(j)
                break
        else:
            if canonical_tournament(rev, n) == canonical_tournament(A, n):
                self_dual.append(i)
                checked.add(i)

    print(f"Self-dual tournaments: {len(self_dual)}")
    print(f"Reversal pairs: {len(reversal_pairs)}")

    # Check if reversal pairs have same sigs at k=n-1
    same_sig_pairs = [(i,j) for i,j in reversal_pairs if sigs[i] == sigs[j]]
    diff_sig_pairs = [(i,j) for i,j in reversal_pairs if sigs[i] != sigs[j]]
    print(f"Reversal pairs with SAME k={n-1} sig: {len(same_sig_pairs)}")
    print(f"Reversal pairs with DIFF k={n-1} sig: {len(diff_sig_pairs)}")

    # Find all pairs with same sig
    sig_to_idx = defaultdict(list)
    for i, sig in sigs.items():
        sig_to_idx[sig].append(i)
    collision_pairs = [(idxs[0], idxs[1]) for idxs in sig_to_idx.values() if len(idxs) >= 2]

    print(f"\nAll k={n-1} collision pairs: {len(collision_pairs)}")
    for i, j in collision_pairs:
        Ai, _ = tourns[i]
        Aj, _ = tourns[j]
        is_rev = (i,j) in reversal_pairs or (j,i) in reversal_pairs
        print(f"  T{i} vs T{j}: reversal pair={is_rev}, scores: {score_sequence(Ai,n)} vs {score_sequence(Aj,n)}")


if __name__ == '__main__':
    # Full reversal analysis first
    for n in [4, 5, 6]:
        full_reversal_analysis(n)

    # Then deep analysis of hard pairs
    for n in [4, 5, 6]:
        analyze_hard_pairs(n)
