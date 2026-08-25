# Counting Revolution V67 — exhaustive n=10 scan of the new component alone

Component:

`vertex_deleted_charpoly_spanning_tree_joint_multiset`

For each vertex `v`, delete it and pair:

- exact adjacency characteristic-polynomial coefficient tuple of `G-v`;
- exact spanning-tree count of `G-v`.

The component is the sorted multiset of the ten pairs.

## Optimized exact implementation

V67 computes the ten vertex-deleted characteristic-polynomial coefficient tuples
simultaneously using an exact integer Faddeev-LeVerrier/adjugate recurrence, rather
than recomputing ten separate characteristic polynomials.

Before the full scan, the optimized implementation is required to match the
published V64 component hashes on all **1,868** V63 collision members and is also
cross-checked against the original direct deletion implementation.

## Exhaustive population

Brendan McKay complete order-10 catalog:

- graphs: **12,005,168**
- connected: **11,716,571**
- compressed SHA-256: `a16f47a95e3e174f4b08042fec95dce8b67712b0e465b5097ffd9334dde2faf8`
- decompressed SHA-256: `923cabf28082cba3ee296251d23eee21b32056b36cf4952e42958d468357df36`

## Exact duplicate logic

A 64-bit deterministic digest is used only as a duplicate filter.

Equal exact component values necessarily serialize identically and therefore have
equal filter digests. Consequently an exact component collision cannot be hidden by
the filter. Every duplicate-digest candidate is recomputed and compared by the full
exact component tuple.

If `exact_collision_groups == 0`, the component alone is collision-free on all
**12,005,168** catalog graphs.

Because the expanded candidate space is the original 13 top-level components plus
this new component, a collision-free single component would establish minimum
nonempty cardinality **1 within that explicit 14-component space**.

## Scope

Counting this rich vertex-deletion multiset as one top-level component is a
candidate-space convention. A minimum of 1 in that space would not mean the
underlying information content is globally minimal, and would not establish a
general reconstruction or graph-isomorphism theorem.
