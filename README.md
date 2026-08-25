# Counting Revolution: Counting-Based Algebraic and Graph Invariants

> **Research status:** computational exploration of counting-valued invariants with provenance-bound exhaustive finite results for order-3 magmas and order-8/order-9/order-10 simple graphs. The repository includes exact top-level component-minimality analyses for the validated order-8 and order-9 graph signatures, plus an order-10 boundary result showing that the original 13-component signature is no longer complete and an augmented five-component witness that restores collision-freedom on the complete order-10 catalog. It does **not** establish a general graph-isomorphism algorithm, an asymptotic scaling theorem, or an externally validated state-of-the-art benchmark.

The central idea is simple: instead of recording only whether a law or structural pattern is satisfied, record **how often** it is satisfied. This produces richer finite signatures for algebraic structures and graphs.

## Experiment family 1: binary operations on small sets

For a magma `(S, *)`, examples of count-valued features include associative-triple counts, commuting-pair counts, identity/zero/idempotency counts, alternative/flexibility counts and selected sorted maps or multisets derived from the operation table.

### Reproducible order-3 result

The order-3 experiment has a public evidence bundle at [`benchmarks/publication-evidence/2026-08-24-v47/`](benchmarks/publication-evidence/2026-08-24-v47/).

The artifact independently reconstructs:

- **19,683** labeled binary operations;
- **3,330** isomorphism classes under `Sym(3)`;
- parity with the source implementation's canonicalization and 29 invariant fields on all 3,330 canonical classes.

The 29 original fields leave one non-isomorphic pair unresolved. The additional invariant

```text
n_left_square_absorption = |{(x,y) : x*x = x*y}|
```

separates that pair. Within the explicit candidate space consisting of the 29 source invariants plus this new invariant, exhaustive subset search finds no complete signature with six or fewer fields and finds a complete seven-field signature.

One verified witness is:

1. `col_multisets`
2. `n_lsd_triples`
3. `left_fixed`
4. `right_fixed`
5. `diagonal`
6. `cube_map_sorted`
7. `n_left_square_absorption`

**Scope:** this is a finite exhaustive result for binary operations on a 3-element set. “Minimum cardinality 7” means minimum only within the stated 30-field candidate space.

## Experiment family 2: graph signatures

The graph experiments combine isomorphism invariants such as degree sequences, closed-walk traces / characteristic-polynomial information, component and distance statistics, spanning-tree counts, local neighborhood statistics and induced-subgraph summaries.

### Reproducible order-8 result

The order-8 full-signature evidence is at [`benchmarks/publication-evidence/2026-08-25-v51/`](benchmarks/publication-evidence/2026-08-25-v51/).

Using Brendan McKay's complete catalog of **12,346 non-isomorphic simple graphs on 8 vertices**, the exact tested full signature produced:

- **12,346 / 12,346** distinct signatures;
- **0** collision groups;
- source-vs-independent full-signature parity on **12,346 / 12,346** graphs;
- exact Matrix-Tree spanning-tree parity on **12,346 / 12,346** graphs;
- **24,692** deterministic relabeling-invariance checks.

**Scope:** this is a finite exhaustive order-8 result only.

### Order-8 top-level component minimality

The order-8 minimality evidence is at [`benchmarks/publication-evidence/2026-08-25-v54/`](benchmarks/publication-evidence/2026-08-25-v54/).

V54 exhaustively checks every subset of the validated **13 top-level components** through cardinality 3:

| Cardinality | Subsets checked | Collision-free subsets |
|---:|---:|---:|
| 1 | 13 | 0 |
| 2 | 78 | 0 |
| 3 | 286 | 13 |

Therefore the minimum collision-free cardinality is **3 within this explicit 13-component space**.

One representative complete triple is:

1. `characteristic_coefficients`
2. `wiener_index`
3. `local_clustering_multiset`

It produces **12,346 / 12,346** distinct signatures.

### Order-9 top-level component minimality

The order-9 evidence is at [`benchmarks/publication-evidence/2026-08-25-v59/`](benchmarks/publication-evidence/2026-08-25-v59/).

Using McKay's complete catalog of **274,668 non-isomorphic simple graphs on 9 vertices**, V59 exhaustively checks all subsets through the first complete cardinality:

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

**Scope:** “minimum cardinality 4” is only within those exact 13 top-level component definitions.

### Order-10 boundary and augmented recovery

The order-10 evidence is at [`benchmarks/publication-evidence/2026-08-25-v64/`](benchmarks/publication-evidence/2026-08-25-v64/).

The exact population is Brendan McKay's complete catalog of **12,005,168 non-isomorphic simple graphs on 10 vertices** (**11,716,571** connected). The raw decompressed catalog is not committed because it is about 114.5 MiB; the evidence binds the official compressed source with exact compressed and decompressed SHA-256 values.

#### V63: the order-9 four-component witness fails at order 10

V63 exhaustively evaluates the order-9 representative four-component witness on all 12,005,168 catalog graphs:

1. `characteristic_coefficients`
2. `spanning_tree_count`
3. `local_clustering_multiset`
4. `neighbor_degree_profile`

Result:

- distinct signatures: **12,004,233 / 12,005,168**;
- exact collision groups: **933**;
- collision members: **1,868**;
- group sizes: **931 pairs + 2 triples**.

Every duplicate-digest candidate was exact-tuple resolved, so the collision conclusion does not depend on hash collision resistance.

#### V64: the original full 13-component signature also fails at order 10

V64 recomputes all original 13 top-level components for every one of the 1,868 V63 collision members and requires exact repository-source parity on **1,868 / 1,868** graphs.

The full original 13-component signature still leaves:

- **264** collision groups;
- **528** graph members;
- all residual groups are non-isomorphic pairs.

Because the V63 four components are a subset of the full 13, every full-13 collision must lie inside V63's exhaustive collision set. These 264 pairs therefore certify that the original 13-component signature is **not complete at order 10**.

#### New deletion-deck component

V64 adds:

```text
vertex_deleted_charpoly_spanning_tree_joint_multiset
```

For every vertex `v`, form `G-v`, compute its exact adjacency characteristic-polynomial coefficients and exact spanning-tree count, pair those two values, and sort the ten pairs as a multiset.

V64 performs **3,736** deterministic relabeling-invariance checks on this new component.

The new component separates every member of every one of V63's **933 / 933** exhaustive four-component collision groups. Therefore the augmented five-component signature

1. `characteristic_coefficients`
2. `spanning_tree_count`
3. `local_clustering_multiset`
4. `neighbor_degree_profile`
5. `vertex_deleted_charpoly_spanning_tree_joint_multiset`

is collision-free on **12,005,168 / 12,005,168** catalog graphs.

This is a sparse exhaustive certificate: any augmented-five collision would already have to collide on its first four components, and V63 exhaustively enumerated all non-singleton equivalence groups under those four components.

**Scope:** this establishes a finite order-10 collision-free five-component witness in an expanded invariant space. It does **not** establish that five is minimal, does not establish completeness for `n > 10`, and does not imply a general graph-isomorphism algorithm or asymptotic growth law.

### Finite order-8 / order-9 / order-10 observation

Within the original 13-component family:

- **order 8:** minimum complete top-level cardinality **3**;
- **order 9:** minimum complete top-level cardinality **4**;
- **order 10:** even the full **13-component** signature is not complete.

After adding the new vertex-deletion component, a **five-component** order-10 witness is collision-free.

These are finite exhaustive observations on three complete catalogs. They do **not** establish monotone or asymptotic growth of the required signature complexity.

## Relation to Weisfeiler–Leman

Specific pairs can witness that one invariant family distinguishes graphs that 1-WL does not. Such examples are finite witnesses, not a general statement that the counting signature is strictly stronger than Weisfeiler–Leman on every graph family or dimension.

## Relation to graph reconstruction

Subgraph and vertex-deletion data are related to graph reconstruction and graph isomorphism. The general Graph Reconstruction Conjecture remains open. The new vertex-deletion invariant used here is therefore presented only as a finite computational invariant, not as a solution to reconstruction or graph isomorphism in general.

## GPU benchmark status

`gpu_fingerprinter.py` explores batch GPU calculation of graph features. Historical README versions reported approximately `2.88M graphs/sec` for an n=8 batch. That number is **not currently supported by a committed raw timing artifact** with exact hardware/software versions, warmup, repetitions, raw timings and correctness output, so it remains a historical local measurement rather than a reproducible benchmark claim.

## What is currently defensible

- order-3: **3,330** isomorphism classes reconstructed from all **19,683** labeled binary operations;
- order-3: minimum **7** within the explicit 30-field candidate space;
- order-8: the validated full signature is collision-free on **12,346 / 12,346** catalog graphs;
- order-8: minimum **3** within the explicit 13 top-level component space;
- order-9: minimum **4** within the same 13 top-level component space;
- order-10: the inherited order-9 four-component witness leaves **933** exact collision groups;
- order-10: the original full 13-component signature leaves **264** non-isomorphic collision pairs;
- order-10: the augmented five-component signature with `vertex_deleted_charpoly_spanning_tree_joint_multiset` is collision-free on **12,005,168 / 12,005,168** catalog graphs;
- these are finite exhaustive results for the exact populations and candidate spaces stated above.

## What is not established here

- a new general graph-isomorphism solution;
- completeness for graph orders greater than 10;
- minimum cardinality of the expanded order-10 invariant space;
- global minimality over arbitrary graph invariants or internal component subfeatures;
- monotone or asymptotic growth of minimum signature complexity with graph order;
- super-exponential asymptotic amplification;
- general strict superiority over Weisfeiler–Leman;
- publication-grade GPU throughput comparisons;
- priority or novelty claims without a dedicated literature review.

## Reproducibility priorities

The order-3, order-8, order-9 and order-10 finite results are now backed by committed evidence and independent verifiers. Remaining priorities are:

1. determine whether minimum cardinality can be established in the **expanded 14-component order-10 candidate space** without requiring an impractical full recomputation of the new deletion-deck component on all 12 million graphs;
2. clean stale claim/provenance language in `graph_n8_exhaustive.py` while preserving immutable evidence snapshots;
3. decompose successful top-level components into finer subfeatures and test whether smaller claim-safe signatures exist;
4. characterize structural commonalities of the order-10 full-13 collision pairs;
5. capture raw GPU benchmark timings with exact hardware/software, warmup, repetitions and correctness protocols before making performance claims.

## Installation

```bash
pip install numpy networkx
```

PyTorch/CUDA is optional for GPU experiments.

## License status

No repository-level `LICENSE` file is currently committed. Until provenance and licensing are explicitly resolved, do not infer reuse rights from earlier README wording.
