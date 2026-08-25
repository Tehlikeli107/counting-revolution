# Counting Revolution — finite n=10 information frontier
## Publication Release Candidate V104

This release candidate freezes the clean, publication-facing evidence layer for
the order-10 deleted-card information-frontier computation.

## Main finite result

For the complete catalog of 12,005,168 non-isomorphic simple graphs on 10
vertices, and the fixed aligned atomic deleted-card family

`{e1,e2,e3,e4,e5,e6,e7,e8,e9,tree}`,

let `m(b)` be the minimum number of atomic fields needed for global
injectivity when an arbitrary auxiliary channel of at most `b` fixed-width bits
is allowed.

The exact result for `b=0,...,6` is:

`m(b) = [5,3,3,2,2,2,1]`.

Important thresholds:

- 1 bit is the first budget allowing a 3-field base.
- 3 bits is the first budget allowing a 2-field base.
- 6 bits is the first budget allowing a 1-field base.

## Directory layout

- `theorem/` — frozen theorem, frontier table, proof dependency map, safe claims.
- `evidence/` — explicit obstruction atlas:
  - 45 atomic pairs × 5 collision witnesses;
  - 10 atomic single fields × 33 collision witnesses.
- `literature/` — preliminary novelty/prior-art audit, claim matrix, references,
  paper outline, release checklist.
- `reproducibility/` — dependency DAG, raw-replication requirements, release
  verifier and machine-readable manifest.
- `certificates/` — exact V101/V102/V103 compact result certificates.

## Verification modes

### A. Certificate-only verification
Run:

`python reproducibility/VERIFY_RELEASE.py`

This does **not** repeat the historical 12M scans. It verifies:

- release file hashes;
- V101/V102/V103 outer hashes and internal manifests;
- frontier vector `[5,3,3,2,2,2,1]`;
- obstruction atlas cardinalities;
- publication-readiness scope/novelty wording.

### B. Raw full-scan replication
This is a heavier historical replication path. See
`reproducibility/RAW_REPLICATION_REQUIREMENTS.json`.

The compact V104 release does not pretend that certificate verification is the
same as independently rerunning every 12M-graph computation.

## Claim scope

This release supports an **exact finite n=10 result inside the explicitly fixed
atomic family**. It does not establish:

- a general graph-isomorphism algorithm or complexity bound;
- the Reconstruction Conjecture;
- the same frontier for n>10;
- novelty or priority over all literature.

The preliminary literature audit found close prior work, especially Dehmer,
Emmert-Streib & Grabner (2014), on complete multivariate discrimination for
connected graphs of orders 9 and 10. The distinctive claim here must therefore
remain the exact restricted-family information frontier and its explicit
certificate architecture, not generic n=10 graph discrimination.
