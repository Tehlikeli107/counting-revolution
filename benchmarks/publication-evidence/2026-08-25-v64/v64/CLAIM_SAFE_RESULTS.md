# Counting Revolution V64 — sparse exact n=10 refinement

V63 exhaustively scanned McKay's complete **12,005,168**-graph
order-10 catalog with:

1. `characteristic_coefficients`
2. `spanning_tree_count`
3. `local_clustering_multiset`
4. `neighbor_degree_profile`

and found **933** exact collision groups containing
**1868** graphs.

V64 does not rerun 12 million graphs. It validates the entire V63 artifact by
manifest/SHA-256, recomputes all **13 source top-level components** for every one of
the 1,868 collision members and requires exact parity with the repository source.

## Full 13-component boundary

Among the V63 collision members, even the full original 13-component signature leaves:

- **264** collision groups;
- **528** graph members;
- all groups are non-isomorphic pairs.

Because the V63 four-component signature is a subset of the original full 13
components, every full-13 collision must occur inside V63's exhaustive collision set.
Therefore these 264 pairs certify that the original full 13-component signature is
**not complete at n=10**.

## New deletion-deck component

V64 adds:

`vertex_deleted_charpoly_spanning_tree_joint_multiset`

For each vertex `v`, delete it and compute on `G-v`:

- exact characteristic-polynomial coefficients;
- exact spanning-tree count.

Sort the ten `(charpoly, spanning-tree)` pairs as a multiset.

This is an isomorphism invariant because an isomorphism bijects deleted vertices and
preserves both invariants of each deleted subgraph.

V64 performs **3,736** deterministic relabeling
checks on the V63 collision members.

## Sparse completeness certificate

The new component distinguishes every member inside every one of V63's
**933 / 933**
representative-4 collision groups.

Therefore the augmented five-component signature:

1. `characteristic_coefficients`
2. `spanning_tree_count`
3. `local_clustering_multiset`
4. `neighbor_degree_profile`
5. `vertex_deleted_charpoly_spanning_tree_joint_multiset`

is collision-free on all **12,005,168 / 12,005,168**
graphs in the complete McKay order-10 catalog.

The proof is sparse but exhaustive: any augmented-five collision would already have
to be a representative-four collision, and V63 enumerated all such collisions.

## Scope

This establishes a finite order-10 upper bound in an **expanded** invariant space.
It does not establish that five is minimal, does not claim the original 13-component
space can classify n=10, and makes no claim for n>10.
