# Order-10 deletion-deck component — exhaustive finite evidence

This directory records V67, an exhaustive scan of the single top-level component

`vertex_deleted_charpoly_spanning_tree_joint_multiset`

on Brendan McKay's complete catalog of **12,005,168 non-isomorphic simple graphs on
10 vertices**.

## Component definition

For each vertex `v`, delete `v`. On the 9-vertex graph `G-v`, compute:

- the exact adjacency characteristic-polynomial coefficient tuple;
- the exact spanning-tree count.

Pair those two values and return the sorted multiset of all ten pairs.

The component is an isomorphism invariant: a graph isomorphism bijects vertices and
maps `G-v` to an isomorphic vertex-deleted subgraph, preserving both paired
invariants.

## Optimized exact implementation

The original V64 implementation directly recomputed ten deleted-graph characteristic
polynomials.

V67 uses an exact integer Faddeev-LeVerrier/adjugate recurrence to obtain all ten
principal vertex-deleted characteristic-polynomial coefficient tuples together.

Before the exhaustive scan, V67 requires:

- optimized component hash == the published V64 value on **1,868 / 1,868** V63
  collision members;
- **256** optimized-vs-direct reference cross-checks.

V67 also records:

- **2,048** direct graph6 parser vs NetworkX checks;
- **2,048** optimized relabeling-invariance checks.

## Exact catalog binding

Official source:

`https://users.cecs.anu.edu.au/~bdm/data/graph10.g6.gz`

Validated population:

- graph records: **12,005,168**
- connected graphs: **11,716,571**
- compressed bytes: **31,112,164**
- compressed SHA-256:
  `a16f47a95e3e174f4b08042fec95dce8b67712b0e465b5097ffd9334dde2faf8`
- decompressed bytes: **120,051,680**
- decompressed SHA-256:
  `923cabf28082cba3ee296251d23eee21b32056b36cf4952e42958d468357df36`

The raw catalog is not embedded in this evidence bundle.

## Exhaustive result

V67 writes one deterministic 64-bit digest per exact component value only as a
duplicate filter.

Equal exact component values necessarily have identical canonical serialization and
therefore identical filter digests. Thus an exact component collision cannot be
hidden by this filter. Any duplicate digest would be recomputed and compared by the
full exact tuple.

Observed over all **12,005,168** catalog graphs:

- duplicate-digest candidate groups: **0**
- duplicate-digest candidate members: **0**
- exact collision groups: **0**
- exact collision members: **0**
- distinct exact component values: **12,005,168 / 12,005,168**

Therefore this single top-level component is collision-free on the complete McKay
order-10 catalog.

## Minimum cardinality in the explicit expanded top-level space

The expanded candidate space consists of:

- the original **13** top-level components;
- this new deletion-deck component.

So it contains **14 top-level components**.

Because one component in this explicit space is already collision-free, the minimum
**nonempty top-level cardinality is 1 within this explicit 14-component space**.

This statement is intentionally narrow. The deletion-deck component is internally
rich and packages ten vertex-deleted graph summaries. A top-level cardinality of 1
does **not** mean globally minimal information content, a one-scalar invariant, a
graph-reconstruction theorem, or a general graph-isomorphism solution.

## Claim boundary

Supported:

> On Brendan McKay's complete order-10 catalog of 12,005,168 non-isomorphic simple
> graphs, `vertex_deleted_charpoly_spanning_tree_joint_multiset` assigns a distinct
> value to every catalog graph. Consequently the minimum nonempty cardinality is 1
> within the explicit 14-component top-level candidate space consisting of the
> original 13 components plus this new component.

Not supported:

- global minimality over subfeatures or arbitrary graph invariants;
- completeness for `n > 10`;
- a proof of the Graph Reconstruction Conjecture;
- a general graph-isomorphism algorithm;
- asymptotic, novelty, or priority claims.
