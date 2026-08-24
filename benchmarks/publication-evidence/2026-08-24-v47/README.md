# Order-3 magma signature — reproducible finite evidence

This directory records a provenance-bound exhaustive computation for **all binary
operations on a 3-element set**.

## Scope

The result is finite and deliberately narrow:

- labeled binary operations: **19,683** (`3^9`);
- isomorphism classes reconstructed under all 6 relabelings: **3,330**;
- candidate search space: the **29 invariant fields already implemented** in
  `complete_classifier.py` at source commit `ed406102a6d048ae3805c48eabecd456f503bcc1`, plus one additional
  count-valued invariant;
- minimum cardinality within that explicit 30-field candidate space: **7**.

This is **not** a theorem about magmas of arbitrary order, **not** a claim that seven
is minimal among every mathematically conceivable invariant, and **not** a novelty
or priority claim.

## Source provenance

Repository: `salihcankurnaz/counting-revolution`  
Branch: `master`  
Source commit: `ed406102a6d048ae3805c48eabecd456f503bcc1`  
`complete_classifier.py` Git blob: `7f295b6dd5c08db2a9914d6e1b3d182ca058fd7e`  
`complete_classifier.py` SHA-256: `d350aeffa193946e63b672ce2a4bf5e1ebfca2140311ed6803d64c1ee32a2870`

The public verifier checks the exact repository HEAD and source blob before
reconstructing the experiment.

## Why an additional invariant is needed

The 29 source invariants leave one pair of non-isomorphic order-3 magmas
indistinguishable. V46/V47 independently verified that the pair agrees on **all 29**
original fields.

The added field is

```text
n_left_square_absorption
= |{(x,y) in S^2 : x*x = x*y}|
```

For the unresolved pair, the values are **6** and **3**.

`candidate_invariance_argument.md` gives the elementary relabeling/isomorphism
argument showing that this count is an isomorphism invariant.

## Minimum-cardinality search within the explicit candidate space

Three pairs of original fields induce identical partitions of the 3,330 classes:

- `n_idempotent` ≡ `sq_fixed_points`
- `n_commuting` ≡ `n_anticommuting`
- `output_freq` ≡ `cayley_indeg`

Thus the 29 original fields induce **26 distinct partitions**.

Because the unresolved pair is equal on every original field and differs on
`n_left_square_absorption`, the new field is mandatory within this 30-field search
space.

The verifier exhaustively checks every subset consisting of the candidate plus up to
five distinct original partitions:

| Total fields | Original partitions chosen | Subsets checked | Complete signatures |
|---:|---:|---:|---:|
| 1 | 0 | 1 | 0 |
| 2 | 1 | 26 | 0 |
| 3 | 2 | 325 | 0 |
| 4 | 3 | 2,600 | 0 |
| 5 | 4 | 14,950 | 0 |
| 6 | 5 | 65,780 | 0 |

A complete signature appears with the candidate plus six original partitions, so
the minimum cardinality **within this explicit candidate set** is **7**.

One verified 7-field witness is:

1. `col_multisets`
2. `n_lsd_triples`
3. `left_fixed`
4. `right_fixed`
5. `diagonal`
6. `cube_map_sorted`
7. `n_left_square_absorption`

`minimal_signature_classes.csv` contains all 3,330 canonical classes and these seven
field values. The seven-field tuples are unique for all 3,330 rows.

## Irredundancy of the displayed witness

Deleting any one field from the displayed seven-field witness destroys completeness:

| Removed field | Distinct classes remaining |
|---|---:|
| `col_multisets` | 3,227 |
| `n_lsd_triples` | 3,324 |
| `left_fixed` | 3,282 |
| `right_fixed` | 3,305 |
| `diagonal` | 3,314 |
| `cube_map_sorted` | 3,327 |
| `n_left_square_absorption` | 3,328 |

This deletion test establishes irredundancy of this witness; the exhaustive smaller-
subset search establishes cardinality 7 as the minimum **within the stated 30-field
search space**.

## Files

- `CLAIM_SAFE_RESULTS.json` — machine-readable result and scope.
- `minimal_signature_classes.csv` — all 3,330 canonical classes and the 7-field witness.
- `minimum_search.csv` — exhaustive no-solution counts through total cardinality 6 and
  the first cardinality-7 solution.
- `partition_equivalences.json` — original fields that induce identical partitions.
- `selected_signature_deletion_test.csv` — one-field deletion test.
- `candidate_invariance_argument.md` — proof of isomorphism invariance for the added count.
- `independent_minimality_verifier_v47.py` — exact verifier used to reconstruct the result.
- `environment.json` — capture environment.
- `methodology.json` — public methodology and claim boundaries.
- `source_hashes.json` — exact source/capture provenance.
- `PUBLIC_MANIFEST.json` — SHA-256/byte manifest for this public artifact.

## Claim-safe interpretation

Supported:

> For all 19,683 binary operations on a 3-element set, exhaustive canonicalization
> yields 3,330 isomorphism classes. Within the explicit candidate space consisting
> of 29 source invariants plus `n_left_square_absorption`, exhaustive subset search
> shows that no signature of six or fewer fields is complete, while a seven-field
> signature distinguishes all 3,330 classes.

Not supported:

- general completeness for larger magmas;
- global minimality over all conceivable invariants;
- asymptotic scaling claims;
- novelty or priority claims;
- publication priority without an external literature review.
