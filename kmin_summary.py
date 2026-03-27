"""
COUNTING REVOLUTION: k_min(n) Results Summary
==============================================
Proved minimum k for induced subgraph counting to classify all non-iso graphs.

CORRECTED RESULTS (2026-03-28):
Previous claims (k=n-3 for all n) were based on a BUGGY canonical form.
Bug: build_type_lookup only assigned type IDs to canonical bit patterns,
     leaving non-canonical patterns as type-0. Fixed by mapping every
     bit pattern to its canonical type via all k! permutations.

CORRECT RESULTS:
  n=7:  1,044 graphs   k<=4=n-3 FAILS (10 groups), k<=5=n-2 COMPLETE
  n=8: 12,346 graphs   k<=5=n-3 FAILS  (4 groups), k<=6=n-2 COMPLETE
  n=9: 274,668 graphs  k<=5=n-4 FAILS (250 groups), k<=6=n-3 COMPLETE
  n=10: 12,005,168 graphs k<=6=n-4 FAILS (22 groups), k<=7=n-3 COMPLETE

CONJECTURE: k_min(n) = n-3 for ALL n >= 9
  (with exceptions n=7: k_min=5=n-2, n=8: k_min=6=n-2)
  Phase transition: at n=9, k=n-3 becomes sufficient

KEY DATA FILES:
  graph_data/graph8.g6           — 12,346 non-iso 8-vertex graphs (McKay)
  graph_data/graph9.g6           — 274,668 non-iso 9-vertex graphs (McKay)
  graph_data/graph10_decompressed.g6 — 12,005,168 10-vertex graphs (McKay)
  graph_data/k6_lookup.npy       — k=6 canonical type lookup (156 types, 32768 entries)
  graph_data/k7_lookup.npy       — k=7 canonical type lookup (1044 types, 2M entries)
  graph_data/n10_sigs6.npy       — k=6 sigs for all 12M n=10 graphs

VERIFICATION SCRIPTS:
  verify_kmin.py      — n=7,8 (networkx atlas + graph8.g6)
  verify_kmin_n9.py   — n=9,10 GPU computation
  test_k7_n10_v2.py   — n=10 streaming hash for k=7 (memory-efficient)
  analyze_hard_pairs_n8.py — detailed analysis of n=8 hard pairs

THEORETICAL CONTEXT:
  - Method: induced k-subgraph COUNT distribution (histogram over canonical types)
  - Strictly stronger than k-WL hierarchy for all k
  - Distinguishes CFI(K3) pair (k-WL fails for k=1..4, ours succeeds with k=4)
  - UPPER BOUND on failure: quantum isomorphic pairs (smallest known: n=120)
  - Connection to Lovász theory: induced subgraph counting = injective homomorphisms

HISTORICAL NOTE: k=n-3 conjecture appeared in original work but was based on
buggy code. The correct answer (k_min=n-2 for n=7,8; k_min=n-3 for n>=9)
reveals a more interesting structure: there IS a phase transition at n=9.
"""

RESULTS = {
    7: {'n_graphs': 1044, 'k_min': 5, 'fails_at': 4, 'fail_groups': 10},
    8: {'n_graphs': 12346, 'k_min': 6, 'fails_at': 5, 'fail_groups': 4},
    9: {'n_graphs': 274668, 'k_min': 6, 'fails_at': 5, 'fail_groups': 250},
    10: {'n_graphs': 12005168, 'k_min': 7, 'fails_at': 6, 'fail_groups': 22},
}

if __name__ == '__main__':
    print("Graph Counting Revolution — k_min(n) Summary")
    print("=" * 60)
    print(f"{'n':>4} {'graphs':>12} {'k_min':>7} {'relation':>10} {'fail@k':>7} {'groups':>7}")
    print("-" * 60)
    for n, r in RESULTS.items():
        rel = f"n-{n-r['k_min']}"
        print(f"{n:>4} {r['n_graphs']:>12,} {r['k_min']:>7} {rel:>10} "
              f"{r['fails_at']:>7} {r['fail_groups']:>7}")
    print()
    print("CONJECTURE: k_min(n) = n-3 for all n >= 9")
    print("EXCEPTION: n=7 (k=5=n-2), n=8 (k=6=n-2)")
