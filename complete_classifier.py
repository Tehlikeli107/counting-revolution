"""
Complete Isomorphism Classifier for |S|=3
==========================================
Goal: Find a MINIMAL set of computable structural invariants that
distinguish ALL 3,330 isomorphism classes of binary operations on {0,1,2}.

If we can do this with a small number of invariants, it would be a genuine
mathematical contribution — a complete classification of all magmas on 3 elements.
"""

import numpy as np
from itertools import product, permutations
from collections import defaultdict, Counter
import json
import time

def enumerate_all_ops(n):
    """Generate all n^(n^2) binary operations on {0,...,n-1}."""
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
    """Canonical form under Sym(n) — lexicographically smallest."""
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

def compute_invariants(op, n):
    """Compute a battery of structural invariants for a binary operation."""
    inv = {}

    # 1. Image size: |{x*y : x,y in S}|
    inv['image_size'] = len(set(op.flatten()))

    # 2. Idempotent count: |{x : x*x = x}|
    inv['n_idempotent'] = sum(1 for a in range(n) if op[a,a] == a)

    # 3. Diagonal multiset (sorted): the values x*x
    inv['diagonal'] = tuple(sorted(op[a,a] for a in range(n)))

    # 4. Left identity count
    inv['n_left_id'] = sum(1 for e in range(n) if all(op[e,a] == a for a in range(n)))

    # 5. Right identity count
    inv['n_right_id'] = sum(1 for e in range(n) if all(op[a,e] == a for a in range(n)))

    # 6. Left zero count (z such that z*a = z for all a)
    inv['n_left_zero'] = sum(1 for z in range(n) if all(op[z,a] == z for a in range(n)))

    # 7. Right zero count (z such that a*z = z for all a)
    inv['n_right_zero'] = sum(1 for z in range(n) if all(op[a,z] == z for a in range(n)))

    # 8. Commutativity count: |{(a,b) : a<b, a*b = b*a}|
    inv['n_commuting'] = sum(1 for a in range(n) for b in range(a+1,n) if op[a,b] == op[b,a])

    # 9. Associativity count: |{(a,b,c) : (a*b)*c = a*(b*c)}|
    inv['n_assoc_triples'] = sum(1 for a in range(n) for b in range(n) for c in range(n)
                                  if op[op[a,b],c] == op[a,op[b,c]])

    # 10. Row multisets: sorted tuple of (sorted row as tuple)
    rows = tuple(sorted(tuple(sorted(op[a,:])) for a in range(n)))
    inv['row_multisets'] = rows

    # 11. Column multisets: sorted tuple of (sorted column as tuple)
    cols = tuple(sorted(tuple(sorted(op[:,a])) for a in range(n)))
    inv['col_multisets'] = cols

    # 12. Row image sizes (sorted): how many distinct outputs per row
    inv['row_image_sizes'] = tuple(sorted(len(set(op[a,:])) for a in range(n)))

    # 13. Column image sizes (sorted)
    inv['col_image_sizes'] = tuple(sorted(len(set(op[:,a])) for a in range(n)))

    # 14. Output frequency distribution: how often each element appears
    freq = sorted(Counter(op.flatten()).values(), reverse=True)
    inv['output_freq'] = tuple(freq)

    # 15. Number of fixed points of left multiplication:
    # For each a, |{b : a*b = b}|, then sort
    inv['left_fixed'] = tuple(sorted(sum(1 for b in range(n) if op[a,b] == b) for a in range(n)))

    # 16. Number of fixed points of right multiplication
    inv['right_fixed'] = tuple(sorted(sum(1 for a in range(n) if op[a,b] == a) for b in range(n)))

    # 17. Automorphism group size
    aut_count = 0
    for perm in permutations(range(n)):
        is_aut = True
        for i in range(n):
            for j in range(n):
                if perm[op[i,j]] != op[perm[i], perm[j]]:
                    is_aut = False
                    break
            if not is_aut:
                break
        if is_aut:
            aut_count += 1
    inv['aut_size'] = aut_count

    # 18. Number of sub-magmas (subsets closed under the operation)
    n_sub = 0
    for mask in range(1, 2**n):  # nonempty subsets
        subset = [i for i in range(n) if mask & (1 << i)]
        closed = True
        for a in subset:
            for b in subset:
                if op[a,b] not in subset:
                    closed = False
                    break
            if not closed:
                break
        if closed:
            n_sub += 1
    inv['n_submagmas'] = n_sub

    # 19. Sub-magma sizes (sorted list)
    sub_sizes = []
    for mask in range(1, 2**n):
        subset = [i for i in range(n) if mask & (1 << i)]
        closed = True
        for a in subset:
            for b in subset:
                if op[a,b] not in subset:
                    closed = False
                    break
            if not closed:
                break
        if closed:
            sub_sizes.append(len(subset))
    inv['sub_sizes'] = tuple(sorted(sub_sizes))

    # 20. Flexibility count: |{(a,b) : a*(b*a) = (a*b)*a}|
    inv['n_flexible'] = sum(1 for a in range(n) for b in range(n)
                            if op[a,op[b,a]] == op[op[a,b],a])

    # 21. Left self-dist count: |{(a,b,c) : a*(b*c) = (a*b)*(a*c)}|
    inv['n_lsd_triples'] = sum(1 for a in range(n) for b in range(n) for c in range(n)
                               if op[a,op[b,c]] == op[op[a,b],op[a,c]])

    # 22. Number of elements in the "center" (commutes with everything AND associates)
    center_count = 0
    for c_elem in range(n):
        in_center = True
        for a in range(n):
            if op[c_elem, a] != op[a, c_elem]:
                in_center = False
                break
            for b in range(n):
                if op[op[c_elem,a],b] != op[c_elem,op[a,b]]:
                    in_center = False
                    break
                if op[op[a,c_elem],b] != op[a,op[c_elem,b]]:
                    in_center = False
                    break
            if not in_center:
                break
        if in_center:
            center_count += 1
    inv['center_size'] = center_count

    # 23. "Cayley graph" in-degree sequence of the left multiplication graph
    # Each element a defines a function L_a: b -> a*b
    # The "Cayley graph" has edges b -> a*b for each a
    # In-degree of element c = |{(a,b) : a*b = c}|
    in_deg = tuple(sorted(sum(1 for a in range(n) for b in range(n) if op[a,b] == c) for c in range(n)))
    inv['cayley_indeg'] = in_deg

    # 24. Row as permutation? (sorted indicator)
    inv['n_latin_rows'] = sum(1 for a in range(n) if len(set(op[a,:])) == n)
    inv['n_latin_cols'] = sum(1 for a in range(n) if len(set(op[:,a])) == n)

    # 25. "Squaring map" structure: a -> a*a
    sq_map = tuple(op[a,a] for a in range(n))
    # Orbit structure of squaring map
    sq_image = len(set(sq_map))
    sq_fixed = sum(1 for a in range(n) if op[a,a] == a)
    inv['sq_image_size'] = sq_image
    inv['sq_fixed_points'] = sq_fixed

    # 26. Cubing map: a -> (a*a)*a
    cube_map = tuple(op[op[a,a], a] for a in range(n))
    inv['cube_map_sorted'] = tuple(sorted(cube_map))

    # 27. Anti-commutativity indicator: how many pairs have a*b != b*a
    inv['n_anticommuting'] = sum(1 for a in range(n) for b in range(a+1,n) if op[a,b] != op[b,a])

    return inv


def main():
    n = 3
    total = n ** (n * n)  # 19683

    print("=" * 70)
    print("  COMPLETE ISOMORPHISM CLASSIFIER FOR |S|=3")
    print("  Goal: minimal invariant set distinguishing all 3,330 iso classes")
    print("=" * 70)

    t0 = time.time()

    # Generate all operations
    print(f"\n  Generating all {total} operations...")
    all_ops = enumerate_all_ops(n)

    # Compute canonical forms (isomorphism classes)
    print("  Computing isomorphism classes...")
    iso_map = {}  # canonical form -> list of operation indices
    op_to_canon = {}
    for idx, op in enumerate(all_ops):
        canon = canonical_form(op, n)
        if canon not in iso_map:
            iso_map[canon] = []
        iso_map[canon].append(idx)
        op_to_canon[idx] = canon

    n_iso = len(iso_map)
    print(f"  {n_iso} isomorphism classes (expected: 3,330)")

    # Pick one representative per iso class
    representatives = {}
    for canon, members in iso_map.items():
        representatives[canon] = members[0]

    # Compute all invariants for representatives
    print(f"\n  Computing invariants for {n_iso} representatives...")
    all_invariants = {}
    inv_keys = None
    for i, (canon, rep_idx) in enumerate(representatives.items()):
        if i % 500 == 0:
            print(f"    {i}/{n_iso}...")
        inv = compute_invariants(all_ops[rep_idx], n)
        all_invariants[canon] = inv
        if inv_keys is None:
            inv_keys = list(inv.keys())

    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s")

    # Now: greedy selection of invariants
    print(f"\n{'='*70}")
    print("  GREEDY INVARIANT SELECTION")
    print(f"{'='*70}")

    # Start with all canons in one class
    canons = list(representatives.keys())

    # For each invariant, compute how many classes it creates
    used_invariants = []
    current_partition = {0: set(range(len(canons)))}  # one big class
    n_classes = 1

    target = n_iso  # 3330

    for step in range(len(inv_keys)):
        best_key = None
        best_gain = 0
        best_n_classes = 0

        for key in inv_keys:
            if key in used_invariants:
                continue

            # Refine current partition by this invariant
            new_partition = {}
            class_id = 0
            for old_class_id, members in current_partition.items():
                sub = defaultdict(set)
                for m in members:
                    val = all_invariants[canons[m]][key]
                    # Make hashable
                    if isinstance(val, np.ndarray):
                        val = tuple(val.flatten())
                    sub[val].add(m)
                for s in sub.values():
                    new_partition[class_id] = s
                    class_id += 1

            n_new = len(new_partition)
            gain = n_new - n_classes
            if gain > best_gain:
                best_gain = gain
                best_key = key
                best_n_classes = n_new
                best_partition = new_partition

        if best_gain == 0:
            break

        used_invariants.append(best_key)
        current_partition = best_partition
        n_classes = best_n_classes
        pct = 100 * n_classes / target
        print(f"  +{best_key}: {n_classes} classes ({pct:.1f}% of {target}) [+{best_gain}]")

        if n_classes >= target:
            break

    print(f"\n  Final: {n_classes} classes using {len(used_invariants)} invariants")
    print(f"  Target: {target} isomorphism classes")

    if n_classes >= target:
        print(f"\n  *** COMPLETE CLASSIFICATION ACHIEVED! ***")
        print(f"  {len(used_invariants)} invariants suffice to distinguish ALL {target} iso classes")
        print(f"\n  Minimal invariant set:")
        for i, key in enumerate(used_invariants):
            print(f"    {i+1}. {key}")
    else:
        print(f"\n  Gap: {target - n_classes} iso classes still merged")
        print(f"  Need additional invariants beyond the {len(inv_keys)} computed")

        # Analyze the remaining merged classes
        merged_count = sum(1 for v in current_partition.values() if len(v) > 1)
        max_merged = max(len(v) for v in current_partition.values())
        print(f"  {merged_count} groups still merged, largest has {max_merged} iso types")

    # Invariant correlation analysis
    print(f"\n{'='*70}")
    print("  INVARIANT STATISTICS")
    print(f"{'='*70}")

    for key in used_invariants[:10]:  # top 10
        vals = [all_invariants[c][key] for c in canons]
        unique = len(set(str(v) for v in vals))
        print(f"  {key}: {unique} distinct values")

    # Save results
    results = {
        'n_iso_classes': n_iso,
        'n_invariants_used': len(used_invariants),
        'invariant_order': used_invariants,
        'n_classes_achieved': n_classes,
        'complete': n_classes >= target,
    }

    with open('theorem_results/classifier_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to theorem_results/classifier_results.json")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
