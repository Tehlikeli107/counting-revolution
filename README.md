# Counting Revolution: Counting-Based Algebraic and Graph Invariants

> **Research status:** computational exploration of counting-valued invariants with provenance-bound exhaustive finite results for order-3 magmas and order-8/order-9 simple graphs. The repository includes exact top-level component-minimality analyses for the validated order-8 and order-9 graph signatures. It does **not** establish a general graph-isomorphism algorithm, an asymptotic scaling theorem, or an externally validated state-of-the-art benchmark.

The central idea is simple: instead of recording only whether a law or pattern is satisfied, record **how often** it is satisfied. This produces richer finite signatures for algebraic structures and graphs.

## Experiment family 1: binary operations on small sets

For a magma `(S, *)`, examples of count-valued features include:

- number of associative triples;
- number of commuting pairs;
- identity/zero/idempotency counts;
- alternative/flexibility counts;
- selected sorted maps or multisets derived from the operation table.

`complete_classifier.py` enumerates all `3^(3^2) = 19,683` binary operations on a 3-element set, canonicalizes them under all relabelings, and searches for a small invariant set that separates the resulting isomorphism classes.

### Reproducible order-3 result

The order-3 experiment has a public, provenance-bound evidence bundle at [`benchmarks/publication-evidence/2026-08-24-v47/`](benchmarks/publication-evidence/2026-08-24-v47/).

The artifact independently reconstructs:

- **19,683** labeled binary operations;
- **3,330** isomorphism classes under `Sym(3)`;
- parity with the source implementation's canonicalization and 29 invariant fields on all 3,330 canonical classes.

The 29 original invariant fields leave one pair of non-isomorphic classes indistinguishable. The added count-valued invariant

```text
n_left_square_absorption = |{(x,y) : x*x = x*y}|
```

takes values **6** and **3** on that unresolved pair.

Within the explicit candidate space consisting of the **29 source invariants plus this one additional invariant**, exhaustive subset search found no complete signature with six or fewer fields and found a complete seven-field signature. One verified witness is:

1. `col_multisets`
2. `n_lsd_triples`
3. `left_fixed`
4. `right_fixed`
5. `diagonal`
6. `cube_map_sorted`
7. `n_left_square_absorption`

**Scope:** this is a finite exhaustive result for binary operations on a 3-element set. “Minimum cardinality 7” means minimum only within the stated 30-field candidate space. It is not a claim of global minimality over all conceivable magma invariants, does not generalize to larger orders, and carries no novelty/priority claim.

### Scaling claim

Historical versions of this README described the amplification from Boolean to counting signatures as **super-exponential** based on experiments at very small set sizes. That asymptotic claim is not established by the current evidence. Finite increases at `|S|=2,3,4` can motivate a scaling hypothesis, but they do not prove an asymptotic growth law.

## Experiment family 2: graph signatures

The graph scripts combine several isomorphism invariants, including examples such as:

- degree sequences;
- closed-walk traces / characteristic-polynomial information;
- component and distance statistics;
- spanning-tree counts;
- local neighborhood statistics;
- induced 4-vertex pattern summaries.

### Reproducible order-8 result

The order-8 experiment has a public, provenance-bound evidence bundle at [`benchmarks/publication-evidence/2026-08-25-v51/`](benchmarks/publication-evidence/2026-08-25-v51/).

Using Brendan McKay's complete catalog of **12,346 non-isomorphic simple graphs on 8 vertices**, the exact tested full signature produced:

- **12,346** catalog graphs;
- **11,117** connected catalog graphs;
- **12,346** distinct full signatures;
- **0** collision groups.

Independent validation in V51 additionally performed:

- independent graph6 decoding and NetworkX round-trip checks for **12,346/12,346** records;
- source-vs-independent full-signature parity for **12,346/12,346** graphs;
- exact-integer Matrix-Tree spanning-tree parity for **12,346/12,346** graphs;
- **24,692** deterministic relabeling-invariance checks;
- an exhaustive 64-labeled-graph sanity check for the induced-4 component's order-4 typing.

**Scope:** this is a finite exhaustive order-8 result. It does not by itself establish completeness at larger graph orders, a general graph-isomorphism algorithm, an asymptotic theorem, or a novelty/priority claim.

### Reproducible order-8 top-level component minimality

A second public evidence bundle at [`benchmarks/publication-evidence/2026-08-25-v54/`](benchmarks/publication-evidence/2026-08-25-v54/) analyzes how many of the **13 top-level components** of the validated V51 signature are needed on the same 12,346-graph catalog.

V54 exhaustively checks every subset of cardinality 1, 2 and 3:

| Cardinality | Subsets checked | Collision-free subsets |
|---:|---:|---:|
| 1 | 13 | 0 |
| 2 | 78 | 0 |
| 3 | 286 | 13 |

Therefore the minimum complete-signature cardinality is **3 within this explicit 13-component top-level search space**.

One compact representative complete triple is:

1. `characteristic_coefficients`
2. `wiener_index`
3. `local_clustering_multiset`

It produces **12,346 / 12,346** distinct reduced signatures on the complete order-8 catalog.

**Scope:** “minimum cardinality 3” means minimum only among the 13 top-level V51 signature components on the complete order-8 catalog. It is not a global minimum over component subfeatures or arbitrary graph invariants.

### Reproducible order-9 top-level component minimality

The order-9 minimum-cardinality experiment has a public, provenance-bound evidence bundle at [`benchmarks/publication-evidence/2026-08-25-v59/`](benchmarks/publication-evidence/2026-08-25-v59/).

Using Brendan McKay's complete catalog of **274,668 non-isomorphic simple graphs on 9 vertices** (**261,080** connected), V59 independently recomputed all 13 top-level components and exhaustively searched every subset through the first complete cardinality:

| Cardinality | Subsets checked | Best distinct signatures | Complete subsets |
|---:|---:|---:|---:|
| 1 | 13 | 247,357 / 274,668 | 0 |
| 2 | 78 | 274,034 / 274,668 | 0 |
| 3 | 286 | 274,664 / 274,668 | 0 |
| 4 | 715 | 274,668 / 274,668 | 6 |

Therefore the minimum complete-signature cardinality is **4 within the exact 13 top-level component space on the complete order-9 catalog**.

One representative minimum witness is:

1. `characteristic_coefficients`
2. `spanning_tree_count`
3. `local_clustering_multiset`
4. `neighbor_degree_profile`

It is collision-free on **274,668 / 274,668** catalog graphs.

V59 also records:

- all **1,092** tested subsets of cardinality 1–4;
- all **6** complete four-component witnesses;
- **24/24** deletion tests becoming incomplete after removal of one component;
- the exact counterexamples left by the strongest three-component signatures;
- **3,215** independent-vs-source full-signature parity checks;
- reconstruction of the earlier order-9 failure of the order-8 representative triple: **458 collision groups / 917 members**.

The order-8 representative triple therefore does **not** remain complete at order 9. The order-9 evidence shows that an additional structural component is necessary within this 13-component family, but the successful minimum witness is not required to contain the exact order-8 representative triple.

**Scope:** “minimum cardinality 4” means minimum only among the exact 13 top-level component definitions on McKay's complete order-9 catalog. It is not a minimum over their internal subfeatures or arbitrary graph invariants, and it does not establish completeness for `n > 9`.

### Finite order-8 vs order-9 observation

Within the same explicit 13-component family, the finite exhaustive minimum changes from:

- **order 8:** minimum top-level cardinality **3**;
- **order 9:** minimum top-level cardinality **4**.

This is a concrete two-order finite observation, not evidence of a monotone or asymptotic growth law. No extrapolation to order 10 or beyond is claimed.

## Relation to Weisfeiler–Leman

A specific pair can witness that one invariant family distinguishes graphs that 1-WL does not. The Shrikhande graph and the 4×4 rook graph are a standard difficult pair for color refinement, and the repository explores additional counts on such examples.

A successful separation of one or more WL-hard pairs is a **finite witness**, not a general statement that the full counting signature is strictly stronger than WL on every graph family or every dimension.

## Relation to graph reconstruction

Subgraph counts are deeply connected to graph-isomorphism and reconstruction theory. However, the general Graph Reconstruction Conjecture remains open. Accordingly, this repository makes no claim that a small fixed collection of polynomial-time counts solves graph isomorphism or reconstruction in general.

## GPU benchmark status

`gpu_fingerprinter.py` explores batch GPU calculation of graph features. Historical README versions reported approximately `2.88M graphs/sec` for an n=8 batch.

That number is **not currently supported by a committed raw timing artifact** with exact hardware, software versions, warmup protocol, repetitions, raw timings, and correctness output. It should be treated as a historical local measurement rather than a reproducible benchmark claim.

## What is currently defensible

- the committed order-3 artifact exhaustively reconstructs **3,330** isomorphism classes from all **19,683** labeled binary operations;
- within its explicit 30-field candidate space, exhaustive search establishes minimum complete-signature cardinality **7** for order 3;
- the committed order-8 graph artifact evaluates the exact tested full signature on all **12,346** non-isomorphic simple graphs in McKay's complete order-8 catalog and finds zero collisions;
- within the 13 top-level components of the order-8 signature, exhaustive subset search establishes minimum collision-free cardinality **3** and identifies **13** complete triples;
- the committed order-9 artifact evaluates the same top-level component family on all **274,668** non-isomorphic simple graphs in McKay's complete order-9 catalog;
- within those 13 top-level components, exhaustive search through cardinality 4 establishes order-9 minimum collision-free cardinality **4** and identifies **6** complete four-component witnesses;
- the strongest order-9 three-component signatures reach **274,664 / 274,668**, so cardinality 3 is insufficient in that explicit search space;
- the order-8 minimum-3 and order-9 minimum-4 values are finite empirical facts for these two complete catalogs, not an asymptotic law;
- counting-valued laws can refine Boolean classifications on finite enumerated structures;
- these finite experiments motivate further study of compact, interpretable structural signatures.

## What is not established here

- a new general graph-isomorphism solution;
- completeness of these graph signatures for orders greater than 9;
- global minimality of the order-8 or order-9 signatures over component subfeatures or arbitrary graph invariants;
- global minimality of the seven-field order-3 signature over all conceivable invariants;
- completeness for magmas of order greater than 3;
- monotone or asymptotic growth of minimum signature cardinality with graph order;
- super-exponential asymptotic amplification;
- general strict superiority over Weisfeiler–Leman;
- publication-grade GPU throughput comparisons;
- priority/novelty claims for counting law-satisfaction values without a dedicated literature review.

## Reproducibility priorities

The order-3 magma result, order-8 graph result, order-8 top-level minimality result, and order-9 top-level minimality result are backed by committed raw evidence and independent verifiers. Remaining priorities are:

1. test order 10 on a complete catalog only if the computational/storage cost remains tractable, otherwise use explicitly scoped adversarial/sampled families;
2. decompose the successful order-8/order-9 top-level components into finer subfeatures and study whether smaller claim-safe finite signatures exist;
3. characterize the exact order-9 counterexamples left by the strongest three-component signatures and compare their structural commonalities;
4. capture raw GPU benchmark timings with exact hardware/software, warmup, repetition and correctness protocols before making performance claims;
5. keep every theorem/result statement explicitly scoped to the finite population actually tested.

## Installation

```bash
pip install numpy networkx
```

PyTorch/CUDA is optional for GPU experiments.

## License status

No repository-level `LICENSE` file is currently committed. Until provenance and licensing are explicitly resolved, do not infer reuse rights from earlier README wording.
