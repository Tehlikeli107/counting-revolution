"""
Counting vs Boolean: The Power of Law Indices
==============================================
Key hypothesis: Replacing boolean "satisfies law X?" with counting
"how many tuples satisfy law X?" captures EXPONENTIALLY more
structural information about magmas.

This tests whether PURE COUNTING invariants (law satisfaction counts)
suffice for complete classification, vs structural invariants.
"""

import numpy as np
from itertools import product, permutations
from collections import defaultdict, Counter
import time

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
    """Pure counting invariants: how many tuples satisfy each equation."""
    inv = {}

    # --- 2-variable laws (counted over all n^2 pairs) ---

    # Commutativity: xy = yx
    inv['cnt_commutative'] = sum(1 for a in range(n) for b in range(n) if op[a,b] == op[b,a])

    # Idempotency: xx = x (counted over n elements)
    inv['cnt_idempotent'] = sum(1 for a in range(n) if op[a,a] == a)

    # Left identity: ea = a
    inv['cnt_left_id'] = sum(1 for e in range(n) for a in range(n) if op[e,a] == a)

    # Right identity: ae = a
    inv['cnt_right_id'] = sum(1 for e in range(n) for a in range(n) if op[a,e] == a)

    # Left zero: za = z
    inv['cnt_left_zero'] = sum(1 for z in range(n) for a in range(n) if op[z,a] == z)

    # Right zero: az = z
    inv['cnt_right_zero'] = sum(1 for z in range(n) for a in range(n) if op[a,z] == z)

    # Left alternative: a(ab) = (aa)b
    inv['cnt_left_alt'] = sum(1 for a in range(n) for b in range(n) if op[a,op[a,b]] == op[op[a,a],b])

    # Right alternative: (ba)a = b(aa)
    inv['cnt_right_alt'] = sum(1 for a in range(n) for b in range(n) if op[op[b,a],a] == op[b,op[a,a]])

    # Flexibility: a(ba) = (ab)a
    inv['cnt_flexible'] = sum(1 for a in range(n) for b in range(n) if op[a,op[b,a]] == op[op[a,b],a])

    # Unipotence: xx = yy (count pairs where xx = yy)
    sq_vals = [op[a,a] for a in range(n)]
    inv['cnt_unipotent_pairs'] = sum(1 for a in range(n) for b in range(n) if sq_vals[a] == sq_vals[b])

    # Constancy: xy = uv (count 4-tuples where outputs equal)
    flat = list(op.flatten())
    inv['cnt_constant_pairs'] = sum(1 for i in range(len(flat)) for j in range(len(flat)) if flat[i] == flat[j])

    # Self-absorption: x(xy) = xy
    inv['cnt_left_absorb'] = sum(1 for a in range(n) for b in range(n) if op[a,op[a,b]] == op[a,b])

    # Right self-absorption: (yx)x = yx
    inv['cnt_right_absorb'] = sum(1 for a in range(n) for b in range(n) if op[op[b,a],a] == op[b,a])

    # Squareness: (xy)(xy) = xy
    inv['cnt_square_idem'] = sum(1 for a in range(n) for b in range(n) if op[op[a,b],op[a,b]] == op[a,b])

    # --- 3-variable laws (counted over all n^3 triples) ---

    # Associativity: (xy)z = x(yz)
    inv['cnt_associative'] = sum(1 for a in range(n) for b in range(n) for c in range(n)
                                  if op[op[a,b],c] == op[a,op[b,c]])

    # Left self-distributivity: x(yz) = (xy)(xz)
    inv['cnt_lsd'] = sum(1 for a in range(n) for b in range(n) for c in range(n)
                          if op[a,op[b,c]] == op[op[a,b],op[a,c]])

    # Right self-distributivity: (yz)x = (yx)(zx)
    inv['cnt_rsd'] = sum(1 for a in range(n) for b in range(n) for c in range(n)
                          if op[op[b,c],a] == op[op[b,a],op[c,a]])

    # Mediality: (xy)(uv) = (xu)(yv) — counted over n^4 but collapsed
    inv['cnt_medial'] = sum(1 for a in range(n) for b in range(n)
                             for c in range(n) for d in range(n)
                             if op[op[a,b],op[c,d]] == op[op[a,c],op[b,d]])

    # Entropic: same as medial but let's add paramedial: (xy)(uv) = (vy)(ux)
    inv['cnt_paramedial'] = sum(1 for a in range(n) for b in range(n)
                                 for c in range(n) for d in range(n)
                                 if op[op[a,b],op[c,d]] == op[op[d,b],op[c,a]])

    # Left Bol: x(y(xz)) = (x(yx))z
    inv['cnt_left_bol'] = sum(1 for x in range(n) for y in range(n) for z in range(n)
                               if op[x,op[y,op[x,z]]] == op[op[x,op[y,x]],z])

    # Right Bol: ((zx)y)x = z((xy)x)
    inv['cnt_right_bol'] = sum(1 for x in range(n) for y in range(n) for z in range(n)
                                if op[op[op[z,x],y],x] == op[z,op[op[x,y],x]])

    # Moufang: (xy)(zx) = (x(yz))x
    inv['cnt_moufang'] = sum(1 for x in range(n) for y in range(n) for z in range(n)
                              if op[op[x,y],op[z,x]] == op[op[x,op[y,z]],x])

    # --- Structural counts ---

    # Image size
    inv['cnt_image_size'] = len(set(op.flatten()))

    # Row surjectivity count (number of rows that are permutations)
    inv['cnt_latin_rows'] = sum(1 for a in range(n) if len(set(op[a,:])) == n)

    # Column surjectivity count
    inv['cnt_latin_cols'] = sum(1 for a in range(n) if len(set(op[:,a])) == n)

    # Number of submagmas
    n_sub = 0
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
            n_sub += 1
    inv['cnt_submagmas'] = n_sub

    # Automorphism count
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
    inv['cnt_automorphisms'] = aut_count

    # Left fixed point count (how many (a,b) pairs with a*b = b)
    inv['cnt_left_fixed'] = sum(1 for a in range(n) for b in range(n) if op[a,b] == b)

    # Right fixed point count (how many (a,b) pairs with a*b = a)
    inv['cnt_right_fixed'] = sum(1 for a in range(n) for b in range(n) if op[a,b] == a)

    return inv


def compute_sorted_invariants(op, n):
    """Invariants that are sorted tuples — richer than pure counts."""
    inv = {}

    # Per-element counts, sorted
    inv['sorted_left_fixed'] = tuple(sorted(sum(1 for b in range(n) if op[a,b] == b) for a in range(n)))
    inv['sorted_right_fixed'] = tuple(sorted(sum(1 for a in range(n) if op[a,b] == a) for b in range(n)))
    inv['sorted_row_images'] = tuple(sorted(len(set(op[a,:])) for a in range(n)))
    inv['sorted_col_images'] = tuple(sorted(len(set(op[:,a])) for a in range(n)))
    inv['sorted_diagonal'] = tuple(sorted(op[a,a] for a in range(n)))
    inv['sorted_cube'] = tuple(sorted(op[op[a,a],a] for a in range(n)))

    # Output frequency distribution
    inv['sorted_output_freq'] = tuple(sorted(Counter(op.flatten()).values(), reverse=True))

    # Per-element associativity counts (sorted)
    inv['sorted_assoc_per_elem'] = tuple(sorted(
        sum(1 for b in range(n) for c in range(n) if op[op[a,b],c] == op[a,op[b,c]])
        for a in range(n)
    ))

    # Per-element LSD counts (sorted)
    inv['sorted_lsd_per_elem'] = tuple(sorted(
        sum(1 for b in range(n) for c in range(n) if op[a,op[b,c]] == op[op[a,b],op[a,c]])
        for a in range(n)
    ))

    # Per-element commutativity (sorted)
    inv['sorted_comm_per_elem'] = tuple(sorted(
        sum(1 for b in range(n) if op[a,b] == op[b,a])
        for a in range(n)
    ))

    # Row multisets (strong structural invariant)
    inv['row_multisets'] = tuple(sorted(tuple(sorted(op[a,:])) for a in range(n)))
    inv['col_multisets'] = tuple(sorted(tuple(sorted(op[:,a])) for a in range(n)))

    return inv


def main():
    n = 3
    total = n ** (n * n)

    print("=" * 70)
    print("  COUNTING VS BOOLEAN: THE POWER OF LAW INDICES")
    print("=" * 70)

    t0 = time.time()
    all_ops = enumerate_all_ops(n)

    # Compute iso classes
    print("\n  Computing isomorphism classes...")
    iso_map = {}
    for idx, op in enumerate(all_ops):
        canon = canonical_form(op, n)
        if canon not in iso_map:
            iso_map[canon] = []
        iso_map[canon].append(idx)

    n_iso = len(iso_map)
    canons = list(iso_map.keys())
    rep_idx = {c: iso_map[c][0] for c in canons}

    print(f"  {n_iso} isomorphism classes")

    # --- Phase 1: PURE COUNTING invariants ---
    print(f"\n{'='*70}")
    print("  PHASE 1: PURE COUNTING INVARIANTS (scalar numbers)")
    print(f"{'='*70}")

    count_invs = {}
    for i, canon in enumerate(canons):
        if i % 500 == 0:
            print(f"  Computing: {i}/{n_iso}...")
        count_invs[canon] = compute_counting_invariants(all_ops[rep_idx[canon]], n)

    count_keys = list(count_invs[canons[0]].keys())
    print(f"  {len(count_keys)} counting invariants computed")

    # Greedy selection
    used = []
    partition = {0: set(range(len(canons)))}
    n_classes = 1

    for step in range(len(count_keys)):
        best_key = None
        best_gain = 0
        best_partition = None
        best_n = 0

        for key in count_keys:
            if key in used:
                continue
            new_part = {}
            cid = 0
            for old_id, members in partition.items():
                sub = defaultdict(set)
                for m in members:
                    sub[count_invs[canons[m]][key]].add(m)
                for s in sub.values():
                    new_part[cid] = s
                    cid += 1
            nn = len(new_part)
            gain = nn - n_classes
            if gain > best_gain:
                best_gain = gain
                best_key = key
                best_n = nn
                best_partition = new_part

        if best_gain == 0:
            break

        used.append(best_key)
        partition = best_partition
        n_classes = best_n
        pct = 100 * n_classes / n_iso
        print(f"  +{best_key}: {n_classes} classes ({pct:.1f}%) [+{best_gain}]")

    count_classes = n_classes
    count_used = list(used)

    print(f"\n  PURE COUNTING: {count_classes}/{n_iso} classes ({100*count_classes/n_iso:.2f}%)")
    print(f"  Using {len(count_used)} invariants")

    if count_classes < n_iso:
        merged = sum(1 for v in partition.values() if len(v) > 1)
        max_m = max(len(v) for v in partition.values())
        print(f"  Still merged: {merged} groups, max size {max_m}")

    # --- Phase 2: Add SORTED invariants ---
    print(f"\n{'='*70}")
    print("  PHASE 2: ADDING SORTED (DISTRIBUTIONAL) INVARIANTS")
    print(f"{'='*70}")

    sorted_invs = {}
    for i, canon in enumerate(canons):
        if i % 500 == 0:
            print(f"  Computing: {i}/{n_iso}...")
        sorted_invs[canon] = compute_sorted_invariants(all_ops[rep_idx[canon]], n)

    sorted_keys = list(sorted_invs[canons[0]].keys())
    print(f"  {len(sorted_keys)} sorted invariants computed")

    # Continue from where counting left off
    for step in range(len(sorted_keys)):
        best_key = None
        best_gain = 0
        best_partition = None
        best_n = 0

        for key in sorted_keys:
            if key in used:
                continue
            new_part = {}
            cid = 0
            for old_id, members in partition.items():
                sub = defaultdict(set)
                for m in members:
                    sub[sorted_invs[canons[m]][key]].add(m)
                for s in sub.values():
                    new_part[cid] = s
                    cid += 1
            nn = len(new_part)
            gain = nn - n_classes
            if gain > best_gain:
                best_gain = gain
                best_key = key
                best_n = nn
                best_partition = new_part

        if best_gain == 0:
            break

        used.append(best_key)
        partition = best_partition
        n_classes = best_n
        pct = 100 * n_classes / n_iso
        print(f"  +{best_key}: {n_classes} classes ({pct:.1f}%) [+{best_gain}]")

        if n_classes >= n_iso:
            break

    print(f"\n  COUNTING + SORTED: {n_classes}/{n_iso} classes ({100*n_classes/n_iso:.2f}%)")

    if n_classes >= n_iso:
        print(f"\n  *** COMPLETE CLASSIFICATION WITH COUNTING + SORTED INVARIANTS ***")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("  SUMMARY: BOOLEAN vs COUNTING vs DISTRIBUTIONAL")
    print(f"{'='*70}")

    # Boolean: just 0/1 for each law
    bool_partition = defaultdict(set)
    for i, canon in enumerate(canons):
        ci = count_invs[canon]
        # Convert counts to booleans
        sig = (
            1 if ci['cnt_associative'] == n**3 else 0,
            1 if ci['cnt_commutative'] == n**2 else 0,
            1 if ci['cnt_lsd'] == n**3 else 0,
            1 if ci['cnt_rsd'] == n**3 else 0,
            1 if ci['cnt_medial'] == n**4 else 0,
            1 if ci['cnt_flexible'] == n**2 else 0,
            1 if ci['cnt_left_alt'] == n**2 else 0,
            1 if ci['cnt_right_alt'] == n**2 else 0,
            1 if ci['cnt_idempotent'] == n else 0,
            1 if ci['cnt_left_id'] >= n else 0,
            1 if ci['cnt_right_id'] >= n else 0,
            1 if ci['cnt_left_zero'] >= n else 0,
            1 if ci['cnt_right_zero'] >= n else 0,
            1 if ci['cnt_image_size'] == 1 else 0,
            1 if ci['cnt_unipotent_pairs'] == n**2 else 0,
            1 if ci['cnt_latin_rows'] == n else 0,
            1 if ci['cnt_latin_cols'] == n else 0,
            1 if ci['cnt_left_bol'] == n**3 else 0,
            1 if ci['cnt_right_bol'] == n**3 else 0,
            1 if ci['cnt_moufang'] == n**3 else 0,
            1 if ci['cnt_paramedial'] == n**4 else 0,
            1 if ci['cnt_left_absorb'] == n**2 else 0,
            1 if ci['cnt_right_absorb'] == n**2 else 0,
        )
        bool_partition[sig].add(i)
    n_bool = len(bool_partition)

    print(f"\n  Boolean properties (23 laws): {n_bool} classes ({100*n_bool/n_iso:.1f}%)")
    print(f"  Counting invariants ({len(count_used)} best):  {count_classes} classes ({100*count_classes/n_iso:.1f}%)")
    print(f"  Counting + Sorted ({len(used)} total):  {n_classes} classes ({100*n_classes/n_iso:.1f}%)")
    print(f"  Isomorphism classes:         {n_iso}")

    # Compression ratio
    print(f"\n  INFORMATION CAPTURE:")
    print(f"    Boolean:     {np.log2(n_bool):.2f} / {np.log2(n_iso):.2f} bits ({100*np.log2(n_bool)/np.log2(n_iso):.1f}%)")
    print(f"    Counting:    {np.log2(count_classes):.2f} / {np.log2(n_iso):.2f} bits ({100*np.log2(count_classes)/np.log2(n_iso):.1f}%)")
    print(f"    Count+Sort:  {np.log2(n_classes):.2f} / {np.log2(n_iso):.2f} bits ({100*np.log2(n_classes)/np.log2(n_iso):.1f}%)")

    # The key finding
    print(f"\n  KEY FINDING: Going from Boolean to Counting multiplied")
    print(f"  classification power by {count_classes/n_bool:.1f}x")
    print(f"  ({n_bool} -> {count_classes} classes)")

    t1 = time.time()
    print(f"\n  Total time: {t1-t0:.1f}s")


if __name__ == '__main__':
    main()
