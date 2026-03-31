"""Quick n=5 tournament hard pair analysis."""
from itertools import combinations, permutations
from collections import defaultdict
import numpy as np

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
                    if (bits >> idx) & 1: new_bits |= (1 << dst)
                else:
                    dst = pair_idx[(pj,pi)]
                    if not ((bits >> idx) & 1): new_bits |= (1 << dst)
            min_bits = min(min_bits, new_bits)
        if min_bits not in canon_to_type:
            canon_to_type[min_bits] = type_id; type_id += 1
        arr[bits] = canon_to_type[min_bits]
    return arr, type_id, pairs

def canonical_tournament(A, n):
    pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    pair_idx = {p: i for i,p in enumerate(pairs)}
    bits0 = sum((1<<idx) for idx,(i,j) in enumerate(pairs) if A[j,i])
    min_bits = bits0
    for perm in permutations(range(n)):
        new_bits = 0
        for idx,(i,j) in enumerate(pairs):
            pi,pj = perm[i],perm[j]
            if pi < pj:
                dst = pair_idx[(pi,pj)]
                if A[j,i]: new_bits |= (1 << dst)
            else:
                dst = pair_idx[(pj,pi)]
                if A[i,j]: new_bits |= (1 << dst)
        min_bits = min(min_bits, new_bits)
    return min_bits

def enum_tournaments(n):
    pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    seen = set(); result = []
    for bits in range(1 << len(pairs)):
        A = np.zeros((n,n), dtype=np.int8)
        for idx,(i,j) in enumerate(pairs):
            if (bits >> idx) & 1: A[j,i] = 1
            else: A[i,j] = 1
        c = canonical_tournament(A, n)
        if c not in seen:
            seen.add(c); result.append(A)
    return result

def compute_sig(A, n, k, lookup, pairs_k):
    counts = np.zeros(int(lookup.max())+1, dtype=np.int32)
    for sub in combinations(range(n), k):
        bits = sum((1<<idx) for idx,(i,j) in enumerate(pairs_k) if A[sub[j],sub[i]])
        counts[lookup[bits]] += 1
    return tuple(counts)

def score_seq(A):
    return tuple(sorted(int(x) for x in A.sum(1)))

print("Building n=5 tournaments...", flush=True)
T5 = enum_tournaments(5)
print(f"n=5: {len(T5)} non-iso tournaments", flush=True)

# Test combined k=2,3,4
lookups = {}
for k in range(2, 5):
    L, nt, pk = build_tournament_lookup(k)
    lookups[k] = (L, nt, pk)

combined_sigs = []
for A in T5:
    csig = []
    for k in range(2, 5):
        L, nt, pk = lookups[k]
        s = compute_sig(A, 5, k, L, pk)
        csig.extend(s)
    combined_sigs.append(tuple(csig))

coll = defaultdict(list)
for i, s in enumerate(combined_sigs):
    coll[s].append(i)
hard = [(s, idxs) for s, idxs in coll.items() if len(idxs) > 1]
print(f"Combined k=2,3,4 collisions: {len(hard)} groups", flush=True)

for s, idxs in hard:
    print(f"\nHard group: indices {idxs}", flush=True)
    for idx in idxs:
        A = T5[idx]
        rev = A.T.copy()
        rev_canon = canonical_tournament(rev, 5)
        # find rev in T5
        rev_idx = None
        for jj, B in enumerate(T5):
            if canonical_tournament(B, 5) == rev_canon:
                rev_idx = jj; break
        print(f"  T{idx}: scores={score_seq(A)}, rev_iso_to=T{rev_idx}", flush=True)

    # Deep structural analysis of first pair
    i, j = idxs[0], idxs[1]
    A, B = T5[i], T5[j]
    rev_A = A.T.copy()
    rev_B = B.T.copy()
    print(f"\n  T{i} adjacency matrix:", flush=True)
    for row in A:
        print(f"    {list(row)}", flush=True)
    print(f"  T{j} adjacency matrix:", flush=True)
    for row in B:
        print(f"    {list(row)}", flush=True)

    # Check: is rev(A) iso B?
    print(f"  rev(T{i}) iso T{j}? {canonical_tournament(rev_A,5)==canonical_tournament(B,5)}", flush=True)
    print(f"  rev(T{j}) iso T{i}? {canonical_tournament(rev_B,5)==canonical_tournament(A,5)}", flush=True)

    # Same score sequence?
    print(f"  Same scores? {score_seq(A)==score_seq(B)}: {score_seq(A)} vs {score_seq(B)}", flush=True)

    # Eigenvalues
    eigA = sorted(np.linalg.eigvals(A.astype(float)).real.round(4))
    eigB = sorted(np.linalg.eigvals(B.astype(float)).real.round(4))
    print(f"  Eigenvalues T{i}: {eigA}", flush=True)
    print(f"  Eigenvalues T{j}: {eigB}", flush=True)
    print(f"  Same eigenvalues? {eigA == eigB}", flush=True)

    # Closed walks tr(A^k)
    for p in range(1, 6):
        Ap = np.linalg.matrix_power(A.astype(float), p)
        Bp = np.linalg.matrix_power(B.astype(float), p)
        print(f"  tr(A^{p})={np.trace(Ap):.2f}, tr(B^{p})={np.trace(Bp):.2f}", flush=True)

print("\nDone.", flush=True)
