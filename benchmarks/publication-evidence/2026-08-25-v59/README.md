# Order-9 graph signature global top-level minimality — reproducible finite evidence

This directory records the V59 exhaustive minimum-cardinality analysis over the
**13 top-level components** of the validated graph signature on Brendan McKay's
complete order-9 simple-graph catalog.

## Population

Captured input:

`https://users.cecs.anu.edu.au/~bdm/data/graph9.g6`

- non-isomorphic simple graphs on 9 vertices: **274,668**
- connected graphs: **261,080**
- captured bytes: **2,197,344**
- SHA-256: `839f67ecc73b1f539128694badebe27adf4f0fb1ee6d0663b7ad9868100d5123`

The exact `graph9.g6` bytes are included in this evidence directory.

## Exact subset search

V59 independently recomputed all 13 top-level components for every catalog graph
and exhaustively checked all subsets through the first complete cardinality:

| Cardinality | Subsets checked | Best distinct signatures | Complete subsets |
|---:|---:|---:|---:|
| 1 | 13 | 247,357 / 274,668 | 0 |
| 2 | 78 | 274,034 / 274,668 | 0 |
| 3 | 286 | 274,664 / 274,668 | 0 |
| 4 | 715 | 274,668 / 274,668 | 6 |

Therefore:

> **The minimum cardinality is 4 within the exact 13 top-level component space on
> McKay's complete order-9 catalog.**

`subset_search_up_to_minimum.csv` contains all **1,092** tested subsets.

## Six minimum witnesses

The six complete four-component subsets are:

1. `traces_A1_to_An` + `spanning_tree_count` + `local_clustering_multiset` + `neighbor_degree_profile`
2. `traces_A1_to_An` + `local_clustering_multiset` + `neighbor_degree_profile` + `edge_common_neighbor_profile`
3. `traces_A1_to_An` + `local_clustering_multiset` + `neighbor_degree_profile` + `nonedge_common_neighbor_profile`
4. `characteristic_coefficients` + `spanning_tree_count` + `local_clustering_multiset` + `neighbor_degree_profile`
5. `characteristic_coefficients` + `local_clustering_multiset` + `neighbor_degree_profile` + `edge_common_neighbor_profile`
6. `characteristic_coefficients` + `local_clustering_multiset` + `neighbor_degree_profile` + `nonedge_common_neighbor_profile`

`traces_A1_to_An` and `characteristic_coefficients` carry equivalent characteristic-
polynomial information for this fixed order, explaining the paired witness structure.

For presentation, a simple representative minimum witness is:

1. `characteristic_coefficients`
2. `spanning_tree_count`
3. `local_clustering_multiset`
4. `neighbor_degree_profile`

It is collision-free on **274,668 / 274,668** catalog graphs.

## Why cardinality 3 is insufficient

The strongest three-component signatures reach **274,664 / 274,668** and leave four
non-isomorphic collision pairs.

The two tied best triples are:

- `traces_A1_to_An + local_clustering_multiset + neighbor_degree_profile`
- `characteristic_coefficients + local_clustering_multiset + neighbor_degree_profile`

`best_incomplete_collision_groups.json` records all four exact graph6 counterexample
pairs.

## Irredundancy

Every one of the six minimum witnesses was tested by deleting each of its four
components.

- minimum witnesses: **6**
- deletion tests: **24**
- deletion tests that remained complete: **0**

Thus every recorded four-component minimum witness is deletion-irredundant.

## Independent/source parity

V59 additionally records:

- **3,215** independent-vs-repository-source full-signature parity checks;
- reconstruction of the previously published V54 representative failure:
  **458 collision groups / 917 members**;
- source parity on all **8** graph members occurring in the best cardinality-3
  collision certificates.

## Files

- `CLAIM_SAFE_RESULTS.json` / `.md` — machine/human V59 result.
- `subset_search_up_to_minimum.csv` — all 1,092 cardinality 1–4 subset results.
- `minimum_subsets.csv` — all six complete four-component witnesses.
- `minimum_subset_deletion_test.csv` — all 24 deletion tests.
- `best_incomplete_collision_groups.json` — exact best-three-component counterexamples.
- `component_summary.csv` — distinct-value count for each top-level component.
- `graph9.g6` — exact captured McKay catalog.
- `independent_graph9_global_minimality_v59.py` — verifier used for the capture.
- `catalog_provenance.json` / `source_provenance.json` / `environment.json` — provenance.
- `capture_manifest_v59.json` — original V59 manifest.
- `methodology.json` / `source_hashes.json` — public method and source binding.
- `PUBLIC_MANIFEST.json` — SHA-256/byte manifest over this public bundle.

## Claim boundary

Supported:

> On Brendan McKay's complete catalog of 274,668 non-isomorphic simple graphs on
> 9 vertices, exhaustive subset search over the exact 13 top-level signature
> components finds no complete subset of cardinality 1, 2, or 3 and finds six
> complete subsets of cardinality 4. The minimum cardinality is therefore 4 within
> that explicit 13-component space.

Not supported:

- minimum over internal subfeatures or all possible graph invariants;
- completeness for graphs with more than 9 vertices;
- a general graph-isomorphism solution;
- asymptotic complexity conclusions;
- novelty or priority claims.
