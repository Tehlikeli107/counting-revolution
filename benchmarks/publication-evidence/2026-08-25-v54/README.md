# Order-8 graph signature component minimality — reproducible finite evidence

This directory records the V54 exhaustive subset analysis of the **13 top-level
components** in the validated V51 order-8 graph signature.

## Result

Population: Brendan McKay's complete catalog of **12,346 non-isomorphic simple
graphs on 8 vertices**, using the already committed V51 per-graph signature table.

V54 checked every top-level subset of cardinality 1, 2, and 3:

| Cardinality | Subsets checked | Complete subsets |
|---:|---:|---:|
| 1 | 13 | 0 |
| 2 | 78 | 0 |
| 3 | 286 | 13 |

Therefore the minimum cardinality is **3 within the explicit 13-component V51
top-level search space**.

This is not a global minimum over component subfeatures or arbitrary graph invariants.

## Best pair

The best two-component signatures reach **12,345 / 12,346** distinct signatures.
The shared unresolved pair is:

- catalog index **266**: `G?AFBo`
- catalog index **348**: `G?B@\`w`

`best_pair_collision.json` records all 13 component values for this pair and shows
which components distinguish it.

## Complete triples

There are **13** collision-free three-component subsets. The complete list is in
`complete_triples.csv`.

The only equivalent top-level component partition found is:

`traces_A1_to_An ≡ characteristic_coefficients`

for this fixed order-8 population.

## Representative compact witness

For public presentation, this bundle designates:

1. `characteristic_coefficients`
2. `wiener_index`
3. `local_clustering_multiset`

as a representative complete triple.

It remains collision-free on all **12,346 / 12,346** catalog graphs.

Among the 13 complete triples, this choice has the smallest mean compact JSON
serialized representation on the V51 table: approximately
**84.9 characters per graph**
(median 85, maximum
102).

That selection criterion is about compact representation only. It does **not** claim
minimum runtime, minimum memory under an optimized encoding, or mathematical
optimality beyond the stated 13 top-level components.

`representative_triple_signatures.csv` contains all 12,346 graph6 identifiers and
the three selected component values.

## Files

- `CLAIM_SAFE_RESULTS.json` / `.md` — V54 machine/human result.
- `subset_search_cardinality_1_to_3.csv` — all 377 tested subsets.
- `complete_triples.csv` — all 13 complete triples.
- `complete_triple_deletion_test.csv` — deletion test for every complete triple.
- `best_pair_collision.json` — exact best-pair collision.
- `component_summary.csv` — distinct-value count for each top-level component.
- `partition_equivalences.json` — equivalent component partitions.
- `representative_triple_signatures.csv` — 12,346-row reduced witness table.
- `representative_selection.json` — representative-selection criterion.
- `independent_component_minimality_v54.py` — exact V54 verifier.
- `source_provenance.json` / `source_hashes.json` — source/capture binding.
- `environment.json` — V54 environment.
- `capture_manifest_v54.json` — original V54 manifest.
- `methodology.json` — public methodology and claim boundaries.
- `PUBLIC_MANIFEST.json` — public payload SHA-256 manifest.

## Claim-safe interpretation

Supported:

> On the committed V51 signature table for McKay's 12,346 non-isomorphic simple
> graphs on 8 vertices, none of the 13 single components or 78 pairs is
> collision-free, while 13 of the 286 triples are collision-free. Thus the minimum
> cardinality is 3 within those 13 top-level components.

Not supported:

- global minimum over all graph invariants or subfeatures;
- completeness for n > 8;
- a general graph-isomorphism algorithm;
- asymptotic, novelty, or priority claims.
