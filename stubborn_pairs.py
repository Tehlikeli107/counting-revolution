"""
Stubborn Pairs Analysis
========================
The 2 pairs of non-isomorphic magmas on |S|=3 that pure counting
invariants (scalar numbers) cannot distinguish.

What makes them special? Why does the cubing map distinguish them?
"""

import numpy as np
from itertools import permutations
from collections import defaultdict, Counter

def enumerate_all_ops(n):
    total = n ** (n * n)
    ops = []
    for idx in range(total):
        op = np.zeros((n, n), dtype=np.int32)
        tmp = idx
        for i in range(n):
            for j in range(n):
                op[i, j] = tmp % n
                tmp //= n
        ops.append(op)
    return ops

def canonical_form(op, n):
    best = tuple(op.flatten())
    for perm in permutations(range(n)):
        new_op = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                new_op[perm[i], perm[j]] = perm[op[i, j]]
        key = tuple(new_op.flatten())
        if key < best:
            best = key
    return best

def compute_counting_invariants(op, n):
    inv = {}
    inv['cnt_commutative'] = sum(1 for a in range(n) for b in range(n) if op[a,b] == op[b,a])
    inv['cnt_idempotent'] = sum(1 for a in range(n) if op[a,a] == a)
    inv['cnt_left_id'] = sum(1 for e in range(n) for a in range(n) if op[e,a] == a)
    inv['cnt_right_id'] = sum(1 for e in range(n) for a in range(n) if op[a,e] == a)
    inv['cnt_left_zero'] = sum(1 for z in range(n) for a in range(n) if op[z,a] == z)
    inv['cnt_right_zero'] = sum(1 for z in range(n) for a in range(n) if op[a,z] == z)
    inv['cnt_left_alt'] = sum(1 for a in range(n) for b in range(n) if op[a,op[a,b]] == op[op[a,a],b])
    inv['cnt_right_alt'] = sum(1 for a in range(n) for b in range(n) if op[op[b,a],a] == op[b,op[a,a]])
    inv['cnt_flexible'] = sum(1 for a in range(n) for b in range(n) if op[a,op[b,a]] == op[op[a,b],a])
    sq_vals = [op[a,a] for a in range(n)]
    inv['cnt_unipotent_pairs'] = sum(1 for a in range(n) for b in range(n) if sq_vals[a] == sq_vals[b])
    flat = list(op.flatten())
    inv['cnt_constant_pairs'] = sum(1 for i in range(len(flat)) for j in range(len(flat)) if flat[i] == flat[j])
    inv['cnt_left_absorb'] = sum(1 for a in range(n) for b in range(n) if op[a,op[a,b]] == op[a,b])
    inv['cnt_right_absorb'] = sum(1 for a in range(n) for b in range(n) if op[op[b,a],a] == op[b,a])
    inv['cnt_square_idem'] = sum(1 for a in range(n) for b in range(n) if op[op[a,b],op[a,b]] == op[a,b])
    inv['cnt_associative'] = sum(1 for a in range(n) for b in range(n) for c in range(n) if op[op[a,b],c] == op[a,op[b,c]])
    inv['cnt_lsd'] = sum(1 for a in range(n) for b in range(n) for c in range(n) if op[a,op[b,c]] == op[op[a,b],op[a,c]])
    inv['cnt_rsd'] = sum(1 for a in range(n) for b in range(n) for c in range(n) if op[op[b,c],a] == op[op[b,a],op[c,a]])
    inv['cnt_medial'] = sum(1 for a in range(n) for b in range(n) for c in range(n) for d in range(n) if op[op[a,b],op[c,d]] == op[op[a,c],op[b,d]])
    inv['cnt_paramedial'] = sum(1 for a in range(n) for b in range(n) for c in range(n) for d in range(n) if op[op[a,b],op[c,d]] == op[op[d,b],op[c,a]])
    inv['cnt_left_bol'] = sum(1 for x in range(n) for y in range(n) for z in range(n) if op[x,op[y,op[x,z]]] == op[op[x,op[y,x]],z])
    inv['cnt_right_bol'] = sum(1 for x in range(n) for y in range(n) for z in range(n) if op[op[op[z,x],y],x] == op[z,op[op[x,y],x]])
    inv['cnt_moufang'] = sum(1 for x in range(n) for y in range(n) for z in range(n) if op[op[x,y],op[z,x]] == op[op[x,op[y,z]],x])
    inv['cnt_image_size'] = len(set(op.flatten()))
    inv['cnt_latin_rows'] = sum(1 for a in range(n) if len(set(op[a,:])) == n)
    inv['cnt_latin_cols'] = sum(1 for a in range(n) if len(set(op[:,a])) == n)
    n_sub = 0
    for mask in range(1, 2**n):
        subset = [i for i in range(n) if mask & (1 << i)]
        closed = True
        for a in subset:
            for b in subset:
                if op[a,b] not in subset:
                    closed = False; break
            if not closed: break
        if closed: n_sub += 1
    inv['cnt_submagmas'] = n_sub
    aut_count = 0
    for perm in permutations(range(n)):
        is_aut = True
        for i in range(n):
            for j in range(n):
                if perm[op[i,j]] != op[perm[i], perm[j]]:
                    is_aut = False; break
            if not is_aut: break
        if is_aut: aut_count += 1
    inv['cnt_automorphisms'] = aut_count
    inv['cnt_left_fixed'] = sum(1 for a in range(n) for b in range(n) if op[a,b] == b)
    inv['cnt_right_fixed'] = sum(1 for a in range(n) for b in range(n) if op[a,b] == a)
    return inv


def main():
    n = 3
    all_ops = enumerate_all_ops(n)

    # Build iso classes
    iso_map = {}
    for idx, op in enumerate(all_ops):
        canon = canonical_form(op, n)
        if canon not in iso_map:
            iso_map[canon] = []
        iso_map[canon].append(idx)

    canons = list(iso_map.keys())
    rep_idx = {c: iso_map[c][0] for c in canons}

    # Compute counting invariants
    count_invs = {}
    for canon in canons:
        count_invs[canon] = compute_counting_invariants(all_ops[rep_idx[canon]], n)

    # Build counting signature
    count_keys = list(count_invs[canons[0]].keys())

    # Group by counting signature
    count_partition = defaultdict(list)
    for i, canon in enumerate(canons):
        sig = tuple(count_invs[canon][k] for k in count_keys)
        count_partition[sig].append(i)

    # Find merged groups
    print("=" * 70)
    print("  STUBBORN PAIRS: Non-isomorphic magmas indistinguishable by counting")
    print("=" * 70)

    merged = [(sig, members) for sig, members in count_partition.items() if len(members) > 1]
    print(f"\n  {len(merged)} merged groups found\n")

    for group_idx, (sig, members) in enumerate(merged):
        print(f"  Group {group_idx + 1}: {len(members)} iso classes merged")

        for m in members:
            canon = canons[m]
            rep = rep_idx[canon]
            op = all_ops[rep]
            orbit_size = len(iso_map[canon])

            print(f"\n    Iso class (orbit size {orbit_size}):")
            print(f"      Cayley table:")
            for i in range(n):
                row = ' '.join(str(op[i,j]) for j in range(n))
                print(f"        {row}")

            # Detailed structure
            sq_map = [op[a,a] for a in range(n)]
            cube_map = [op[op[a,a],a] for a in range(n)]
            fourth = [op[op[op[a,a],a],a] for a in range(n)]

            print(f"      Squaring:  {sq_map}")
            print(f"      Cubing:    {cube_map}")
            print(f"      Fourth:    {fourth}")

            # Sorted versions
            print(f"      Sq sorted: {sorted(sq_map)}")
            print(f"      Cu sorted: {sorted(cube_map)}")
            print(f"      4th sorted: {sorted(fourth)}")

            # Left multiplication maps
            for a in range(n):
                L_a = [op[a,b] for b in range(n)]
                print(f"      L_{a}: {L_a}")

            # Right multiplication maps
            for a in range(n):
                R_a = [op[b,a] for b in range(n)]
                print(f"      R_{a}: {R_a}")

            # Check anti-isomorphism (opposite magma)
            opp = np.zeros((n,n), dtype=np.int32)
            for i in range(n):
                for j in range(n):
                    opp[i,j] = op[j,i]
            opp_canon = canonical_form(opp, n)

            if opp_canon == canon:
                print(f"      Self-opposite: YES")
            else:
                print(f"      Self-opposite: NO (opp is different iso class)")

        # Check if the pair are opposites of each other
        c1, c2 = canons[members[0]], canons[members[1]]
        op1 = all_ops[rep_idx[c1]]
        op2 = all_ops[rep_idx[c2]]

        opp1 = np.zeros((n,n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                opp1[i,j] = op1[j,i]
        opp1_canon = canonical_form(opp1, n)

        if opp1_canon == c2:
            print(f"\n    *** These two are OPPOSITE MAGMAS of each other! ***")
            print(f"    (One is the transpose/mirror of the other)")
        else:
            print(f"\n    These are NOT opposite magmas")

    # The deeper question: WHY can't counts distinguish these?
    print(f"\n{'='*70}")
    print("  ANALYSIS: Why counting fails for these pairs")
    print(f"{'='*70}")

    print(f"""
  A counting invariant cnt_P = |{{tuples : P holds}}| is preserved under
  BOTH isomorphism AND anti-isomorphism (transposing the Cayley table).

  If two non-isomorphic magmas are related by anti-isomorphism
  (one is the transpose of the other), then ALL counting invariants
  must agree on them, because:
    cnt_P(M) = cnt_P(M^op) for any equational property P

  This is because for each tuple (a1,...,ak) satisfying P in M,
  there's a corresponding tuple satisfying the "mirror" of P in M^op,
  and for symmetric laws like associativity, commutativity, etc.,
  the mirror is the same law (or another law we're already counting).

  PREDICTION: The stubborn pairs are precisely the non-self-dual
  magmas — those not isomorphic to their opposite.
  """)

    # Verify: how many non-self-dual iso classes exist?
    non_self_dual = 0
    non_self_dual_pairs = []
    seen = set()
    for canon in canons:
        if canon in seen:
            continue
        op = all_ops[rep_idx[canon]]
        opp = np.zeros((n,n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                opp[i,j] = op[j,i]
        opp_canon = canonical_form(opp, n)
        if opp_canon != canon:
            non_self_dual += 1
            if opp_canon not in seen:
                non_self_dual_pairs.append((canon, opp_canon))
                seen.add(canon)
                seen.add(opp_canon)

    print(f"  Non-self-dual iso classes: {non_self_dual}")
    print(f"  Non-self-dual PAIRS (M, M^op): {len(non_self_dual_pairs)}")

    # Which of these are NOT already distinguished by counting?
    undistinguished = 0
    for c1, c2 in non_self_dual_pairs:
        sig1 = tuple(count_invs[c1][k] for k in count_keys)
        sig2 = tuple(count_invs[c2][k] for k in count_keys)
        if sig1 == sig2:
            undistinguished += 1

    print(f"  Non-self-dual pairs NOT distinguished by counting: {undistinguished}")
    print(f"  Non-self-dual pairs already distinguished: {len(non_self_dual_pairs) - undistinguished}")

    # The key insight
    print(f"\n  CONCLUSION:")
    if undistinguished == len(merged):
        print(f"  All {len(merged)} stubborn groups are anti-isomorphic pairs!")
        print(f"  Counting invariants fail EXACTLY for the chirality distinction.")
        print(f"  The cubing map a -> (a*a)*a breaks this symmetry because")
        print(f"  it mixes left and right multiplication asymmetrically.")
    else:
        print(f"  {undistinguished}/{len(merged)} stubborn groups are anti-isomorphic pairs")

    # What about LEFT vs RIGHT counting?
    print(f"\n{'='*70}")
    print("  CHIRALITY: LEFT vs RIGHT counting invariants")
    print(f"{'='*70}")

    # Add inherently asymmetric counts
    for group_idx, (sig, members) in enumerate(merged):
        print(f"\n  Group {group_idx + 1}:")
        for m in members:
            canon = canons[m]
            op = all_ops[rep_idx[canon]]

            # Left-specific counts
            cnt_lxa_eq_a = sum(1 for a in range(n) for x in range(n) if op[x,a] == a)
            cnt_axl_eq_a = sum(1 for a in range(n) for x in range(n) if op[a,x] == a)
            # These are cnt_left_id and cnt_right_id — they're symmetric under transpose

            # ASYMMETRIC: a*(a*a) vs (a*a)*a
            left_cube = tuple(sorted(op[a, op[a,a]] for a in range(n)))
            right_cube = tuple(sorted(op[op[a,a], a] for a in range(n)))

            # a*(a*(a*a)) vs ((a*a)*a)*a
            left_4 = tuple(sorted(op[a, op[a, op[a,a]]] for a in range(n)))
            right_4 = tuple(sorted(op[op[op[a,a], a], a] for a in range(n)))

            print(f"    Canon {m}: L-cube={left_cube}, R-cube={right_cube}, "
                  f"L-4th={left_4}, R-4th={right_4}")


if __name__ == '__main__':
    main()
