# Counting Revolution V59 — order-9 global top-level component minimality

Population: Brendan McKay's complete **274,668** non-isomorphic simple
graphs on 9 vertices.

Search space: the exact **13 top-level components** of the validated counting
signature.

V59 recomputes all 13 components independently for every graph, factors each
component into exact equality partitions, and checks subsets in increasing
cardinality.

## Result

Minimum cardinality within the explicit 13-component space:

**4**

Minimum witness(es) recorded:

- `traces_A1_to_An` + `spanning_tree_count` + `local_clustering_multiset` + `neighbor_degree_profile`
- `traces_A1_to_An` + `local_clustering_multiset` + `neighbor_degree_profile` + `edge_common_neighbor_profile`
- `traces_A1_to_An` + `local_clustering_multiset` + `neighbor_degree_profile` + `nonedge_common_neighbor_profile`
- `characteristic_coefficients` + `spanning_tree_count` + `local_clustering_multiset` + `neighbor_degree_profile`
- `characteristic_coefficients` + `local_clustering_multiset` + `neighbor_degree_profile` + `edge_common_neighbor_profile`
- `characteristic_coefficients` + `local_clustering_multiset` + `neighbor_degree_profile` + `nonedge_common_neighbor_profile`

A previously validated five-component witness is independently rechecked as
**274,668/274,668** distinct.

## Search coverage

- cardinality 1: 13 subsets checked; best 247,357/274,668
- cardinality 2: 78 subsets checked; best 274,034/274,668
- cardinality 3: 286 subsets checked; best 274,664/274,668
- cardinality 4: 715 subsets checked; best 274,668/274,668

## Parity/provenance

- McKay `graph9.g6` SHA-256: `839f67ecc73b1f539128694badebe27adf4f0fb1ee6d0663b7ad9868100d5123`
- published V54 representative failure reconstructed:
  **458 collision groups / 917 members**
- independent-vs-source parity checks: **3,215**
- all best-incomplete collision members additionally source-checked:
  **8**

## Claim boundary

“Minimum” means minimum only among these 13 top-level component definitions on the
complete order-9 catalog. It does not mean minimum over internal subfeatures or all
possible graph invariants, and it does not imply completeness for n>9.
