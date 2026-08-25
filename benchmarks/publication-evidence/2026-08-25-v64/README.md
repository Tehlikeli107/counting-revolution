# Order-10 graph-signature boundary and augmented refinement — reproducible finite evidence

This directory combines two validated stages over Brendan McKay's complete catalog
of **12,005,168 non-isomorphic simple graphs on 10 vertices**.

- `v63/` — exhaustive scan of the four-component witness inherited from the n=9 result.
- `v64/` — exact sparse refinement of every V63 collision member.

## Catalog binding

Official compressed source:

`https://users.cecs.anu.edu.au/~bdm/data/graph10.g6.gz`

Validated capture:

- compressed bytes: **31,112,164**
- compressed SHA-256:
  `a16f47a95e3e174f4b08042fec95dce8b67712b0e465b5097ffd9334dde2faf8`
- decompressed bytes: **120,051,680**
- decompressed SHA-256:
  `923cabf28082cba3ee296251d23eee21b32056b36cf4952e42958d468357df36`
- graph records: **12,005,168**
- connected graphs: **11,716,571**

The raw decompressed `graph10.g6` is not committed because it is about 114.5 MiB.
The official compressed source and both compressed/decompressed hashes bind the exact
population used by V63.

## V63: the n=9 minimum witness stops being complete

V63 exhaustively evaluates:

1. `characteristic_coefficients`
2. `spanning_tree_count`
3. `local_clustering_multiset`
4. `neighbor_degree_profile`

on all **12,005,168** catalog graphs.

Result:

- distinct four-component signatures: **12,004,233**
- exact collision groups: **933**
- exact collision members: **1,868**
- group sizes: **931 pairs + 2 triples**

V63 uses a disk-backed deterministic 64-bit digest only as a duplicate filter.
Every duplicate-digest candidate is recomputed and compared by the full exact tuple,
so the scientific collision decision does not rely on hash collision resistance.

The scan also records **2,048** direct graph6/NetworkX/source parity checks.

## V64: the original full 13-component signature also stops being complete

V64 validates the exact V63 ZIP and recomputes all original 13 top-level components
for **every one of the 1,868 V63 collision members**, requiring repository-source
parity on **1,868 / 1,868** graphs.

The full original 13-component signature still leaves:

- **264** collision groups
- **528** graph members
- all residual groups are non-isomorphic pairs

This is sufficient to show that the original full 13-component signature is not
complete on the order-10 catalog. Any full-13 collision must lie inside the V63
four-component collision set because those four components are a subset of the 13.

## V64 new component

V64 adds:

`vertex_deleted_charpoly_spanning_tree_joint_multiset`

For every vertex `v`, form `G-v` and compute:

- exact adjacency characteristic-polynomial coefficients;
- exact spanning-tree count.

The new component is the sorted multiset of those ten
`(characteristic_coefficients, spanning_tree_count)` pairs.

An isomorphism bijects deleted vertices and maps corresponding vertex-deleted
subgraphs isomorphically, so the multiset is an isomorphism invariant.

V64 checks two deterministic relabelings for each of the 1,868 collision members:
**3,736 relabeling-invariance checks**.

## Sparse exhaustive completeness certificate

The new component distinguishes every member inside every one of V63's
**933 / 933** exhaustive four-component collision groups.

Therefore the augmented five-component signature

1. `characteristic_coefficients`
2. `spanning_tree_count`
3. `local_clustering_multiset`
4. `neighbor_degree_profile`
5. `vertex_deleted_charpoly_spanning_tree_joint_multiset`

is collision-free on **12,005,168 / 12,005,168** catalog graphs.

This conclusion is exhaustive even though V64 computes the new component only on
1,868 graphs: any augmented-five collision would first have to collide on its first
four components, and V63 exhaustively enumerates every non-singleton equivalence
group under those four components.

## Claim boundary

Supported:

> The inherited four-component n=9 witness is not complete at n=10; the original
> full 13-component signature is also not complete at n=10; adding the stated
> vertex-deletion joint multiset to the four-component witness yields a
> collision-free five-component signature on McKay's complete order-10 catalog.

Not supported:

- minimum cardinality in the expanded invariant space;
- completeness for `n > 10`;
- a general graph-isomorphism algorithm;
- asymptotic growth of the required number of components;
- novelty or priority claims.
