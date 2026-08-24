# Counting Revolution: Counting-Based Algebraic and Graph Invariants

> **Research status:** computational exploration of counting-valued invariants with provenance-bound exhaustive finite results for order-3 magmas and order-8 simple graphs, plus additional exploratory graph/GPU experiments. The repository does **not** establish a general graph-isomorphism algorithm, super-exponential scaling theorem, or externally validated state-of-the-art benchmark.

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

The raw 3,330-class table, minimum-search log, deletion test, independent verifier, environment, source hashes, methodology and SHA-256 manifest are committed with the evidence bundle.

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

The order-8 experiment now has a public, provenance-bound evidence bundle at [`benchmarks/publication-evidence/2026-08-25-v51/`](benchmarks/publication-evidence/2026-08-25-v51/).

Using Brendan McKay's complete catalog of **12,346 non-isomorphic simple graphs on 8 vertices**, the exact tested signature produced:

- **12,346** catalog graphs;
- **11,117** connected catalog graphs;
- **12,346** distinct full signatures;
- **0** collision groups.

The capture records the exact `graph8.g6` input (`86,422` bytes, SHA-256 `546a249902101c97d3aa590f93e53366854bd0a6f405aa59bdb32d25c57f845a`) and includes attribution/provenance for the McKay catalog.

Independent validation in V51 additionally performed:

- independent graph6 decoding and NetworkX round-trip checks for **12,346/12,346** records;
- source-vs-independent full-signature parity for **12,346/12,346** graphs;
- exact-integer Matrix-Tree spanning-tree parity for **12,346/12,346** graphs;
- **24,692** deterministic relabeling-invariance checks;
- an exhaustive 64-labeled-graph sanity check showing that the degree-sequence key used by the induced-4 component distinguishes all 11 simple-graph isomorphism types on four vertices.

The evidence directory contains the complete raw 12,346-row signature table, exact empty collision-group record, captured catalog bytes, independent verifier, source/environment/catalog provenance, methodology and SHA-256 manifest.

**Scope:** the supported statement is that the exact tested counting signature is collision-free on McKay's complete catalog of non-isomorphic simple graphs on **8 vertices**. This is a finite exhaustive order-8 result. It does **not** establish completeness for `n > 8`, a general graph-isomorphism algorithm, an asymptotic theorem, or a novelty/priority claim.

## Relation to Weisfeiler–Leman

A specific pair can witness that one invariant family distinguishes graphs that 1-WL does not. The Shrikhande graph and the 4×4 rook graph are a standard difficult pair for color refinement, and the repository explores additional counts on such examples.

A successful separation of one or more WL-hard pairs is a **finite witness**, not a general statement that the full counting signature is strictly stronger than WL on every graph family or every dimension.

## Relation to graph reconstruction

Subgraph counts are deeply connected to graph-isomorphism and reconstruction theory. However, the general Graph Reconstruction Conjecture remains open: even whether the multiset of all `(n-1)`-vertex-deleted subgraphs determines every finite simple graph is not settled in full generality.

Accordingly, this repository makes no claim that a small fixed collection of polynomial-time counts solves graph isomorphism or reconstruction in general.

## GPU benchmark status

`gpu_fingerprinter.py` explores batch GPU calculation of graph features. Historical README versions reported approximately `2.88M graphs/sec` for an n=8 batch.

That number is **not currently supported by a committed raw timing artifact** with exact hardware, software versions, warmup protocol, repetitions, raw timings, and correctness output. It should be treated as a historical local measurement rather than a reproducible benchmark claim.

## What is currently defensible

- the committed order-3 artifact exhaustively reconstructs 3,330 isomorphism classes from all 19,683 labeled binary operations;
- within its explicit 30-field candidate space, exhaustive search establishes minimum complete-signature cardinality 7 for order 3;
- the committed order-8 graph artifact evaluates the exact tested signature on all 12,346 non-isomorphic simple graphs in McKay's complete order-8 catalog and finds 12,346 distinct signatures with zero collisions;
- the order-8 source signature matches an independent implementation on all catalog graphs, with exact spanning-tree and relabeling cross-checks;
- counting-valued laws can refine Boolean classifications on finite enumerated structures;
- selected induced-subgraph counts can distinguish graph pairs that simpler invariants may merge;
- these finite experiments motivate further study of compact, interpretable structural signatures.

## What is not established here

- a new general graph-isomorphism solution;
- completeness of the listed graph signature for orders greater than 8;
- global minimality of the seven-field order-3 signature over all conceivable invariants;
- completeness for magmas of order greater than 3;
- super-exponential asymptotic amplification;
- general strict superiority over Weisfeiler–Leman;
- publication-grade GPU throughput comparisons;
- priority/novelty claims for counting law-satisfaction values without a dedicated literature review.

## Reproducibility priorities

The order-3 magma and order-8 graph results are now backed by committed raw evidence and independent verifiers. Remaining priorities are:

1. reproduce and artifact additional finite graph-order experiments before extending any completeness statement;
2. isolate which signature components are actually necessary on the order-8 catalog and test reduced signatures;
3. investigate collision behavior on larger graph orders using complete catalogs where feasible and carefully sampled/adversarial families otherwise;
4. capture raw GPU benchmark timings with exact hardware/software, warmup, repetition and correctness protocols before making performance claims;
5. keep every theorem/result statement explicitly scoped to the finite population actually tested.

## Installation

```bash
pip install numpy networkx
```

PyTorch/CUDA is optional for GPU experiments.

## License status

No repository-level `LICENSE` file is currently committed. Until provenance and licensing are explicitly resolved, do not infer reuse rights from earlier README wording.
