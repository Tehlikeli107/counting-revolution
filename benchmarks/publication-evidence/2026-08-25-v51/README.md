# Order-8 graph signature — reproducible finite evidence

This directory records a provenance-bound exhaustive collision test for the exact
counting signature implemented in `graph_n8_exhaustive.py`.

## Result

Using Brendan McKay's complete catalog of **12,346 non-isomorphic simple graphs on
8 vertices**, the tested signature produced:

- catalog graphs: **12,346**
- connected catalog graphs: **11,117**
- distinct full signatures: **12,346**
- collision groups: **0**

The result is therefore collision-free on the complete order-8 catalog.

## Input provenance

Catalog:

`https://users.cecs.anu.edu.au/~bdm/data/graph8.g6`

Catalog index:

`https://users.cecs.anu.edu.au/~bdm/data/graphs.html`

Captured `graph8.g6`:

- bytes: **86,422**
- SHA-256: `546a249902101c97d3aa590f93e53366854bd0a6f405aa59bdb32d25c57f845a`
- graph6 records: **12,346**

The catalog index independently lists 12,346 total simple graphs and 11,117 connected
simple graphs on 8 vertices.

Brendan McKay's combinatorial-data collection states that, except where data was
compiled by someone else, McKay releases the data files in the collection under
**CC BY 4.0**. See `CATALOG_ATTRIBUTION.md` and `catalog_provenance.json`.

## Source provenance

Repository: `salihcankurnaz/counting-revolution`  
Source commit: `6dac78defc7614f6d8e534cd78a25ae889e03784`  
`graph_n8_exhaustive.py` Git blob: `8ddf73479a7653bed7b7a93beada9f364d57c9dd`

The exact repository source is not duplicated here as the authoritative source remains
the Git commit above; `source_provenance.json` binds the files used by the capture.

## Independent validation

V51 performed all of the following:

1. decoded all 12,346 graph6 records independently;
2. checked the independent decoder against NetworkX and graph6 round-trip for every record;
3. recomputed the full signature independently for all 12,346 graphs and compared it
   with the repository implementation;
4. recomputed Matrix-Tree spanning-tree counts with exact integer Bareiss elimination
   for all 12,346 graphs and compared them with the source result;
5. applied two fixed nontrivial vertex relabelings to every graph, giving **24,692**
   relabeling-invariance checks;
6. exhaustively checked all 64 labeled 4-vertex simple graphs and confirmed that the
   degree-sequence key used by the induced-4 component distinguishes the 11 order-4
   isomorphism types;
7. grouped all full signatures exactly and found **0 collision groups**.

## Raw evidence

- `graph_signatures.csv` — one row per catalog graph, including graph6 text and full
  serialized signature.
- `collision_groups.json` — exact collision groups; empty for this run.
- `graph8.g6` — exact captured catalog bytes.
- `CLAIM_SAFE_RESULTS.json` / `.md` — machine- and human-readable result.
- `independent_graph8_verifier_v51.py` — verifier used by the capture.
- `catalog_provenance.json` / `catalog_http_metadata.json` — input provenance.
- `source_provenance.json` — exact Git commit/blob/source hashes.
- `environment.json` — execution environment.
- `capture_manifest_v51.json` — original V51 capture manifest.
- `methodology.json` — public method and claim boundaries.
- `PUBLIC_MANIFEST.json` — SHA-256/byte manifest over the public payload.

## Claim-safe interpretation

Supported:

> On Brendan McKay's complete catalog of 12,346 non-isomorphic simple graphs on
> 8 vertices, the exact tested counting signature assigns a distinct signature to
> every catalog graph.

Because the fields were also tested under deterministic relabelings and the source
implementation was cross-checked against an independent implementation, this is strong
finite computational evidence for the exact order-8 experiment.

Not supported:

- completeness for graphs with more than 8 vertices;
- a general graph-isomorphism algorithm;
- an asymptotic complexity theorem derived from this finite result;
- novelty or priority;
- any historical GPU throughput claim.

## Encoding note

All text and CSV files in this public bundle are stored with canonical **LF line
endings** before SHA-256 calculation. This avoids the Windows CRLF normalization issue
encountered during the earlier order-3 publication step.
