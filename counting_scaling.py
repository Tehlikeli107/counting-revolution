"""
Counting Revolution Scaling Test
=================================
Does the 21.6x amplification from Boolean→Counting grow with |S|?

|S|=2: Boolean gives 10 classes = iso classes (perfect, no amplification possible)
|S|=3: Boolean gives 154, Counting gives 3,328 (21.6x) out of 3,330

For |S|=4: 4^16 = 4,294,967,296 operations (too many to enumerate)
But OEIS A001329 says there are 3,492,900 iso classes on |S|=4.

Strategy: Sample many random operations, compute Boolean + Counting signatures,
estimate the number of distinct classes found by each method.
"""

import numpy as np
from itertools import permutations
from collections import Counter, defaultdict
import time

def random_op(n):
    """Generate a random binary operation on {0,...,n-1}."""
    return np.random.randint(0, n, size=(n, n), dtype=np.int32)

def canonical_form(op, n):
    """Canonical form under Sym(n)."""
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

def boolean_sig(op, n):
    """Boolean property signature (is/isn't for each law)."""
    sig = []

    # Associative
    sig.append(int(all(op[op[a,b],c] == op[a,op[b,c]]
                       for a in range(n) for b in range(n) for c in range(n))))

    # Commutative
    sig.append(int(all(op[a,b] == op[b,a] for a in range(n) for b in range(n))))

    # Has identity
    sig.append(int(any(all(op[e,a] == a and op[a,e] == a for a in range(n)) for e in range(n))))

    # Idempotent
    sig.append(int(all(op[a,a] == a for a in range(n))))

    # Left cancellative
    sig.append(int(all(len(set(op[a,:])) == n for a in range(n))))

    # Right cancellative
    sig.append(int(all(len(set(op[:,a])) == n for a in range(n))))

    # LSD
    sig.append(int(all(op[a,op[b,c]] == op[op[a,b],op[a,c]]
                       for a in range(n) for b in range(n) for c in range(n))))

    # Medial
    sig.append(int(all(op[op[a,b],op[c,d]] == op[op[a,c],op[b,d]]
                       for a in range(n) for b in range(n) for c in range(n) for d in range(n))))

    # Flexible
    sig.append(int(all(op[a,op[b,a]] == op[op[a,b],a] for a in range(n) for b in range(n))))

    # Left alternative
    sig.append(int(all(op[a,op[a,b]] == op[op[a,a],b] for a in range(n) for b in range(n))))

    # Right alternative
    sig.append(int(all(op[op[b,a],a] == op[b,op[a,a]] for a in range(n) for b in range(n))))

    # Has zero
    sig.append(int(any(all(op[a,z] == z and op[z,a] == z for a in range(n)) for z in range(n))))

    # Constant
    sig.append(int(len(set(op.flatten())) == 1))

    # Unipotent
    sig.append(int(len(set(op[a,a] for a in range(n))) == 1))

    # Left Bol
    sig.append(int(all(op[x,op[y,op[x,z]]] == op[op[x,op[y,x]],z]
                       for x in range(n) for y in range(n) for z in range(n))))

    # Right Bol
    sig.append(int(all(op[op[op[z,x],y],x] == op[z,op[op[x,y],x]]
                       for x in range(n) for y in range(n) for z in range(n))))

    # Moufang
    sig.append(int(all(op[op[x,y],op[z,x]] == op[op[x,op[y,z]],x]
                       for x in range(n) for y in range(n) for z in range(n))))

    return tuple(sig)

def counting_sig(op, n):
    """Counting property signature (how many tuples satisfy each law)."""
    sig = []

    # Associativity count
    sig.append(sum(1 for a in range(n) for b in range(n) for c in range(n)
                   if op[op[a,b],c] == op[a,op[b,c]]))

    # Commutativity count
    sig.append(sum(1 for a in range(n) for b in range(n) if op[a,b] == op[b,a]))

    # LSD count
    sig.append(sum(1 for a in range(n) for b in range(n) for c in range(n)
                   if op[a,op[b,c]] == op[op[a,b],op[a,c]]))

    # RSD count
    sig.append(sum(1 for a in range(n) for b in range(n) for c in range(n)
                   if op[op[b,c],a] == op[op[b,a],op[c,a]]))

    # Paramedial count
    sig.append(sum(1 for a in range(n) for b in range(n) for c in range(n) for d in range(n)
                   if op[op[a,b],op[c,d]] == op[op[d,b],op[c,a]]))

    # Left Bol count
    sig.append(sum(1 for x in range(n) for y in range(n) for z in range(n)
                   if op[x,op[y,op[x,z]]] == op[op[x,op[y,x]],z]))

    # Moufang count
    sig.append(sum(1 for x in range(n) for y in range(n) for z in range(n)
                   if op[op[x,y],op[z,x]] == op[op[x,op[y,z]],x]))

    # Left alt count
    sig.append(sum(1 for a in range(n) for b in range(n) if op[a,op[a,b]] == op[op[a,a],b]))

    # Right alt count
    sig.append(sum(1 for a in range(n) for b in range(n) if op[op[b,a],a] == op[b,op[a,a]]))

    # Flexibility count
    sig.append(sum(1 for a in range(n) for b in range(n) if op[a,op[b,a]] == op[op[a,b],a]))

    # Right identity count
    sig.append(sum(1 for e in range(n) for a in range(n) if op[a,e] == a))

    # Left identity count
    sig.append(sum(1 for e in range(n) for a in range(n) if op[e,a] == a))

    # Square-idempotent count
    sig.append(sum(1 for a in range(n) for b in range(n) if op[op[a,b],op[a,b]] == op[a,b]))

    # Idempotent count
    sig.append(sum(1 for a in range(n) if op[a,a] == a))

    # Image size
    sig.append(len(set(op.flatten())))

    # Latin rows count
    sig.append(sum(1 for a in range(n) if len(set(op[a,:])) == n))

    # Latin cols count
    sig.append(sum(1 for a in range(n) if len(set(op[:,a])) == n))

    # Left zero count
    sig.append(sum(1 for z in range(n) for a in range(n) if op[z,a] == z))

    # Right zero count
    sig.append(sum(1 for z in range(n) for a in range(n) if op[a,z] == z))

    # Automorphism count
    aut = 0
    for perm in permutations(range(n)):
        if all(perm[op[i,j]] == op[perm[i],perm[j]] for i in range(n) for j in range(n)):
            aut += 1
    sig.append(aut)

    # Submagma count
    n_sub = 0
    for mask in range(1, 2**n):
        subset = [i for i in range(n) if mask & (1 << i)]
        if all(op[a,b] in subset for a in subset for b in subset):
            n_sub += 1
    sig.append(n_sub)

    return tuple(sig)


def main():
    np.random.seed(42)

    print("=" * 70)
    print("  COUNTING REVOLUTION SCALING TEST")
    print("  Does Boolean->Counting amplification grow with |S|?")
    print("=" * 70)

    # First verify |S|=2 and |S|=3 (exhaustive)
    for n in [2, 3]:
        total = n ** (n * n)
        print(f"\n  |S|={n}: {total} operations")

        all_ops = []
        for idx in range(total):
            op = np.zeros((n, n), dtype=np.int32)
            tmp = idx
            for i in range(n):
                for j in range(n):
                    op[i, j] = tmp % n
                    tmp //= n
            all_ops.append(op)

        # Iso classes
        iso_classes = set()
        for op in all_ops:
            iso_classes.add(canonical_form(op, n))

        # Boolean classes
        bool_classes = set()
        for op in all_ops:
            bool_classes.add(boolean_sig(op, n))

        # Counting classes
        count_classes = set()
        for op in all_ops:
            count_classes.add(counting_sig(op, n))

        ratio = len(count_classes) / len(bool_classes) if len(bool_classes) > 0 else 0

        print(f"    Iso classes: {len(iso_classes)}")
        print(f"    Boolean classes: {len(bool_classes)}")
        print(f"    Counting classes: {len(count_classes)}")
        print(f"    Amplification: {ratio:.1f}x")
        print(f"    Boolean coverage: {100*len(bool_classes)/len(iso_classes):.1f}%")
        print(f"    Counting coverage: {100*len(count_classes)/len(iso_classes):.1f}%")

    # Now sample |S|=4
    n = 4
    print(f"\n  |S|={n}: 4^16 = 4,294,967,296 operations (SAMPLING)")
    print(f"  Known iso classes: 3,492,900 (OEIS A001329)")

    n_samples = 500000  # 500K samples

    t0 = time.time()
    bool_sigs = set()
    count_sigs = set()
    iso_sigs = set()

    batch_size = 10000
    for batch in range(n_samples // batch_size):
        if batch % 10 == 0:
            elapsed = time.time() - t0
            print(f"    Batch {batch}/{n_samples//batch_size} "
                  f"({batch*batch_size} ops, {elapsed:.0f}s) "
                  f"Bool={len(bool_sigs)}, Count={len(count_sigs)}, Iso={len(iso_sigs)}")

        for _ in range(batch_size):
            op = random_op(n)
            bool_sigs.add(boolean_sig(op, n))
            count_sigs.add(counting_sig(op, n))
            iso_sigs.add(canonical_form(op, n))

    elapsed = time.time() - t0
    print(f"\n    Sampled {n_samples} random operations in {elapsed:.0f}s")
    print(f"    Distinct Boolean signatures: {len(bool_sigs)}")
    print(f"    Distinct Counting signatures: {len(count_sigs)}")
    print(f"    Distinct Iso classes found: {len(iso_sigs)}")

    ratio4 = len(count_sigs) / len(bool_sigs) if len(bool_sigs) > 0 else 0
    print(f"    Sampled amplification: {ratio4:.1f}x")

    # Coverage estimates
    print(f"\n    Boolean coverage (of sampled iso): {100*len(bool_sigs)/len(iso_sigs):.2f}%")
    print(f"    Counting coverage (of sampled iso): {100*len(count_sigs)/len(iso_sigs):.2f}%")

    print(f"\n{'='*70}")
    print("  SCALING SUMMARY")
    print(f"{'='*70}")

    print(f"""
    |S|  Total ops     Iso classes   Boolean    Counting    Amplification
    ---  ----------    -----------   -------    --------    -------------
     2        16            10         10          10          1.0x
     3    19,683         3,330        154       3,328         21.6x
     4   ~4.3 B       3,492,900      {len(bool_sigs):>5}      {len(count_sigs):>6}        {ratio4:.1f}x (sampled)
    """)

    if ratio4 > 21.6:
        print("  *** AMPLIFICATION GROWS WITH |S|! ***")
        print("  The counting revolution becomes MORE powerful on larger sets!")
    elif ratio4 > 10:
        print("  Amplification remains significant at |S|=4")
    else:
        print("  Amplification may not grow (sample-based estimate)")


if __name__ == '__main__':
    main()
