# Counting Revolution V63 — exhaustive n=10 representative-4 scan

Tested signature:

1. `characteristic_coefficients`
2. `spanning_tree_count`
3. `local_clustering_multiset`
4. `neighbor_degree_profile`

Input: Brendan McKay's complete catalog of **12,005,168** non-isomorphic
simple graphs on 10 vertices.

The official compressed catalog is bound by SHA-256:

`a16f47a95e3e174f4b08042fec95dce8b67712b0e465b5097ffd9334dde2faf8`

The fully decompressed stream is independently rebound after the compute pass:

`923cabf28082cba3ee296251d23eee21b32056b36cf4952e42958d468357df36`

## Duplicate detection

V63 does not retain 12 million Python signature objects. It writes one deterministic
64-bit digest per exact four-component signature to a disk-backed array, sorts those
digests and treats duplicate digests only as **candidates**.

This does not rely on hash collision resistance for the scientific conclusion:

- equal exact signatures necessarily produce equal deterministic digests;
- therefore every exact signature collision must appear in the duplicate-digest set;
- every duplicate-digest candidate is recomputed and compared using the full exact
  Python tuple;
- only exact tuple equality is reported as a real signature collision.

`digest_candidate_groups.json` records the filter candidates and
`exact_collision_groups.json` records the exact resolution.

## Scope

This stage answers only whether this specific four-component witness is
collision-free on the complete n=10 catalog. It does not establish the global
minimum number of top-level components at n=10.
