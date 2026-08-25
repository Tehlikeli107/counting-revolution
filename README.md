# Counting Revolution: Counting-Based Algebraic and Graph Invariants

> **Research status:** computational exploration of counting-valued invariants with provenance-bound exhaustive finite results for order-3 magmas and order-8/order-9/order-10 simple graphs. The repository contains validated order-8/order-9 component-minimality results, order-10 boundary and deletion-deck results, and a frozen finite-order atomic information frontier for the aligned `{e1,...,e9,tree}` deleted-card family. It does **not** establish a general graph-isomorphism algorithm, the Graph Reconstruction Conjecture, an asymptotic scaling law, or a novelty/priority claim.

The central idea is to record **how often** structural laws or patterns occur, rather than only whether they occur. This produces richer finite signatures for algebraic structures and graphs.

## Experiment family 1: binary operations on small sets

### Reproducible order-3 result

Evidence: [`benchmarks/publication-evidence/2026-08-24-v47/`](benchmarks/publication-evidence/2026-08-24-v47/)

The order-3 artifact reconstructs:

- **19,683** labeled binary operations;
- **3,330** isomorphism classes under `Sym(3)`;
- parity with the source implementation's canonicalization and 29 original invariant fields.

The 29 original fields leave one non-isomorphic pair unresolved. The additional invariant

```text
n_left_square_absorption = |{(x,y) : x*x = x*y}|
```

separates that pair.

Within the explicit candidate space consisting of the 29 source invariants plus this new invariant, exhaustive subset search finds no complete signature with six or fewer fields and finds a complete seven-field signature. One verified witness is:

1. `col_multisets`
2. `n_lsd_triples`
3. `left_fixed`
4. `right_fixed`
5. `diagonal`
6. `cube_map_sorted`
7. `n_left_square_absorption`

**Scope:** minimum cardinality 7 means minimum only within the stated 30-field candidate space for binary operations on a 3-element set.

## Experiment family 2: graph signatures

The graph experiments use isomorphism invariants such as degree sequences, closed-walk traces / characteristic-polynomial information, component and distance statistics, spanning-tree counts, neighborhood profiles and induced-subgraph summaries.

### Reproducible order-8 full-signature result

Evidence: [`benchmarks/publication-evidence/2026-08-25-v51/`](benchmarks/publication-evidence/2026-08-25-v51/)

Using Brendan McKay's complete catalog of **12,346 non-isomorphic simple graphs on 8 vertices**, the validated full signature produces:

- **12,346 / 12,346** distinct signatures;
- **0** collision groups;
- source-vs-independent full-signature parity on **12,346 / 12,346** graphs;
- exact Matrix-Tree spanning-tree parity on **12,346 / 12,346** graphs;
- **24,692** deterministic relabeling-invariance checks.

### Order-8 top-level component minimality

Evidence: [`benchmarks/publication-evidence/2026-08-25-v54/`](benchmarks/publication-evidence/2026-08-25-v54/)

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

### Order-9 top-level component minimality

Evidence: [`benchmarks/publication-evidence/2026-08-25-v59/`](benchmarks/publication-evidence/2026-08-25-v59/)

Using McKay's complete catalog of **274,668 non-isomorphic simple graphs on 9 vertices**, V59 exhaustively searches every subset through the first complete cardinality:

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

### Order-10 boundary for the original component family

Evidence: [`benchmarks/publication-evidence/2026-08-25-v64/`](benchmarks/publication-evidence/2026-08-25-v64/)

The population is McKay's complete catalog of **12,005,168 non-isomorphic simple graphs on 10 vertices** (**11,716,571** connected).

#### V63: the order-9 four-component witness fails

The order-9 representative four-component witness produces:

- **12,004,233 / 12,005,168** distinct signatures;
- **933** exact collision groups;
- **1,868** collision members;
- group sizes **931 pairs + 2 triples**.

#### V64: the original full 13-component signature also fails

V64 recomputes all original 13 top-level components for all 1,868 V63 collision members with **1,868 / 1,868** source parity.

The full original 13-component signature still leaves:

- **264** collision groups;
- **528** graph members;
- all residual groups are non-isomorphic pairs.

Because the four V63 components are a subset of the 13, every full-13 collision must occur inside V63's exhaustive collision set. These pairs therefore certify that the original 13-component signature is **not complete at order 10**.

#### V64: augmented five-component recovery

V64 adds:

```text
vertex_deleted_charpoly_spanning_tree_joint_multiset
```

For every vertex `v`, form `G-v`, compute its exact adjacency characteristic-polynomial coefficients and exact spanning-tree count, pair those values, and sort the ten pairs as a multiset.

The new component separates every member of all **933 / 933** V63 collision groups. Therefore the augmented five-component signature is collision-free on **12,005,168 / 12,005,168** catalog graphs.

### Order-10 single-component result

Evidence: [`benchmarks/publication-evidence/2026-08-25-v67/`](benchmarks/publication-evidence/2026-08-25-v67/)

V67 asks a stronger finite question: is the new deletion-deck component itself already collision-free on the complete order-10 catalog?

The implementation computes the ten principal vertex-deleted characteristic-polynomial coefficient tuples with an exact integer Faddeev-LeVerrier/adjugate recurrence, and validates the optimized implementation against the previously published V64 values.

Validation includes:

- optimized-vs-published V64 parity: **1,868 / 1,868**;
- optimized-vs-direct reference cross-checks: **256**;
- graph6 parser-vs-NetworkX checks: **2,048**;
- optimized relabeling-invariance checks: **2,048**.

Exhaustive result over all **12,005,168** order-10 catalog graphs:

- duplicate-digest candidate groups: **0**;
- duplicate-digest candidate members: **0**;
- exact collision groups: **0**;
- exact collision members: **0**;
- distinct exact component values: **12,005,168 / 12,005,168**.

Therefore `vertex_deleted_charpoly_spanning_tree_joint_multiset` is collision-free by itself on the complete McKay order-10 catalog.

The explicit expanded top-level candidate space contains the original 13 components plus this new component, for **14 top-level components** total. Since one component in this space is already collision-free, the minimum **nonempty top-level cardinality is 1 within this explicit 14-component space**.

**Important scope:** the deletion-deck component is internally rich: it packages ten vertex-deleted graph summaries. A top-level cardinality of 1 does **not** mean globally minimal information content, a one-scalar invariant, a proof of the Graph Reconstruction Conjecture, or a general graph-isomorphism solution.

### Order-10 atomic information frontier

Publication release candidate: [`benchmarks/publication-evidence/2026-08-26-v104/`](benchmarks/publication-evidence/2026-08-26-v104/)

The collision-free order-10 deletion-deck component is internally rich, so the
next question is not merely whether it distinguishes the catalog, but how much
of its internal information is actually necessary.

For each deleted card `G-v`, the V104 release fixes the aligned atomic family

```text
e1, e2, e3, e4, e5, e6, e7, e8, e9, tree
```

where `e1..e9` are exact adjacency characteristic-polynomial elementary
symmetric coefficients and `tree` is the exact spanning-tree count. For a
chosen field subset, values belonging to the same deleted card remain paired
before the ten card records are sorted as a multiset.

On the complete catalog of **12,005,168** non-isomorphic simple graphs on
10 vertices, let `m(b)` be the minimum number of these atomic fields required
for global injectivity when an arbitrary deterministic auxiliary channel of at
most `b` fixed-width bits is allowed.

The exact finite-order frontier for `b = 0,...,6` is:

| Auxiliary bits `b` | Exact minimum atomic fields `m(b)` | One matching upper-bound base |
|---:|---:|---|
| 0 | **5** | one of five inclusion-minimal complete 5-field subsets |
| 1 | **3** | `e2 + e4 + e6` plus the explicit V94 one-bit channel |
| 2 | **3** | `e2 + e4 + tree` (maximum base class 4) |
| 3 | **2** | `e2 + e6` (maximum base class 7) |
| 4 | **2** | `e2 + e4` (maximum base class 10) |
| 5 | **2** | `e2 + e4` (maximum base class 10) |
| 6 | **1** | `e4` (maximum base class 57) |

Therefore:

```text
m(0..6) = [5, 3, 3, 2, 2, 2, 1]
```

The matching lower bounds include an explicit obstruction atlas:

- all **45** atomic pairs have five pairwise non-isomorphic order-10 graphs
  with the same aligned pair signature, ruling out arbitrary 1-bit and 2-bit
  repair of every pair;
- all **10** single atomic fields have 33 pairwise non-isomorphic graphs with
  the same single-field signature, directly ruling out arbitrary 3-bit,
  4-bit and 5-bit repair of every single field;
- stronger single-field analysis shows the unique best single atomic field is
  `e4`, whose exact auxiliary optimum is **57 states / 6 fixed-width bits**.

V104 also separates **certificate verification** from **raw full-catalog
replication** and includes a theorem dependency map, explicit obstruction
tables, a preliminary literature/novelty audit and conservative claim wording.

**Scope:** this is an exact result only for the complete finite `n=10` catalog
and the stated aligned atomic family. It does not prove a general
graph-isomorphism algorithm, the Reconstruction Conjecture, the same frontier
for larger orders, or novelty/priority relative to all literature.

### Literature positioning of the new frontier

The V103/V104 literature audit records important prior art. In particular,
full vertex-deck reconstruction is already known computationally well beyond
order 10, polynomial/spectral reconstruction is a classical research area, and
prior work has already reported complete multivariate invariant discrimination
on exhaustive connected order-9/order-10 graph populations.

Accordingly, the defensible contribution here is the **specific finite-order
information frontier for the fixed aligned deleted-card atomic family, with
matching lower/upper certificates and an explicit obstruction atlas**. The
release does not make a "first", "world-first", general graph-isomorphism or
general reconstruction claim.

### Finite order-8 / order-9 / order-10 observation

Within the **original 13-component family**:

- **order 8:** minimum complete top-level cardinality **3**;
- **order 9:** minimum complete top-level cardinality **4**;
- **order 10:** even the full **13-component** signature is not complete.

After expanding the candidate space with the richer deletion-deck component, that new component alone is collision-free at order 10.

These are finite exhaustive observations on complete catalogs. They do **not** establish a monotone or asymptotic law for signature complexity.

## Relation to Weisfeiler–Leman

Specific graph pairs can witness that one invariant family distinguishes examples that 1-WL does not. Such examples are finite witnesses, not a general statement that the counting signatures are strictly stronger than Weisfeiler–Leman on every graph family or dimension.

## Relation to graph reconstruction

Vertex-deletion data are related to graph reconstruction and graph isomorphism. The general Graph Reconstruction Conjecture remains open. The deletion-deck component here is therefore presented only as a finite computational invariant on the tested catalog.

## GPU benchmark status

`gpu_fingerprinter.py` explores batch GPU calculation of graph features. Historical README versions reported approximately `2.88M graphs/sec` for an n=8 batch. That value is not currently supported by a committed raw timing artifact with a complete reproducibility protocol, so it remains a historical local measurement rather than a publication-grade benchmark claim.

## What is currently defensible

- order-3: **3,330** isomorphism classes reconstructed from all **19,683** labeled binary operations;
- order-3: minimum **7** within the explicit 30-field candidate space;
- order-8: the validated full graph signature is collision-free on **12,346 / 12,346** catalog graphs;
- order-8: minimum **3** within the explicit original 13 top-level graph-component space;
- order-9: minimum **4** within that same original 13-component space;
- order-10: the inherited order-9 four-component witness leaves **933** exact collision groups;
- order-10: the original full 13-component signature leaves **264** non-isomorphic collision pairs;
- order-10: the augmented five-component signature is collision-free on **12,005,168 / 12,005,168** graphs;
- order-10: the new deletion-deck component alone is collision-free on **12,005,168 / 12,005,168** graphs;
- order-10: minimum nonempty top-level cardinality is **1 within the explicit expanded 14-component space**;
- order-10 atomic deletion-deck family: exact auxiliary-bit/field-count frontier
  **`[5, 3, 3, 2, 2, 2, 1]` for bit budgets `0..6`** on the complete
  12,005,168-graph catalog;
- order-10: **3 bits** is the first auxiliary-bit budget that permits a
  two-field atomic base (`e2+e6`, exact maximum class 7);
- order-10: **6 bits** is the first auxiliary-bit budget that permits a one-field atomic base
  (`e4`, exact maximum class 57);
- order-10: explicit obstruction certificates cover all **45 atomic pairs**
  (five non-isomorphic common-signature graphs each) and all **10 single
  atomic fields** (33 non-isomorphic common-signature graphs each);
- all of these statements are finite results for the exact populations and candidate spaces stated above.

## What is not established here

- a new general graph-isomorphism solution;
- a proof of the Graph Reconstruction Conjecture;
- completeness for graph orders greater than 10;
- global minimality over arbitrary graph invariants or internal component subfeatures;
- globally minimal information content for the order-10 deletion-deck invariant;
- monotone or asymptotic growth of minimum signature complexity with graph order;
- super-exponential asymptotic amplification;
- general strict superiority over Weisfeiler–Leman;
- publication-grade GPU throughput comparisons;
- novelty or priority claims without a dedicated literature review.

## Reproducibility priorities

The order-3, order-8, order-9 and order-10 finite results are backed by
committed or frozen evidence and independent verifiers. After the V104
information-frontier release candidate, the main priorities are:

1. publish the clean V104 finite-`n=10` frontier evidence without rewriting
   earlier immutable evidence snapshots;
2. complete a broader systematic literature review before any novelty or
   priority wording stronger than "direct prior match not identified in the
   searched literature";
3. convert the frozen proof dependency map into manuscript-ready mathematical
   notation and proofs of the fast deleted-card coefficient identities;
4. search for a more natural mask-free realization of the explicit V94
   one-bit auxiliary channel;
5. keep certificate-only verification and raw 12M-graph replication as
   explicitly separate reproducibility modes;
6. perform larger-order experiments only after a specific theorem question and
   feasible provenance-bound population are defined.

## Installation

```bash
pip install numpy networkx
```

PyTorch/CUDA is optional for GPU experiments.

## License status

No repository-level `LICENSE` file is currently committed. Until provenance and licensing are explicitly resolved, do not infer reuse rights from earlier README wording.
