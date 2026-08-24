# Counting Revolution V51 — order-8 graph validation

## Input

Brendan McKay's `graph8.g6` catalog, captured from:

`https://users.cecs.anu.edu.au/~bdm/data/graph8.g6`

The catalog index lists **12,346** non-isomorphic simple graphs on 8 vertices.

Captured catalog SHA-256:

`546a249902101c97d3aa590f93e53366854bd0a6f405aa59bdb32d25c57f845a`

## Validation performed

- independent graph6 decoder vs NetworkX decoder/round-trip for all 12,346 graphs;
- exact source commit/blob provenance;
- repository `compute_counting_signature` vs an independently implemented signature
  for all 12,346 catalog graphs;
- Matrix-Tree spanning-tree count independently recomputed with integer Bareiss
  elimination for all 12,346 graphs;
- two nontrivial deterministic vertex relabelings checked for every graph
  (24,692 signature-invariance checks);
- exhaustive 4-vertex sanity check showing that the degree-sequence key used by the
  `sub4` component distinguishes all 11 isomorphism classes on 4 vertices;
- exact full-signature collision grouping across the complete catalog.

## Result

Distinct signatures: **12,346 / 12,346**  
Collision groups: **0**

The tested signature is collision-free on the complete McKay order-8 catalog.

## Claim boundary

If collision-free, the supported statement is:

> On Brendan McKay's complete catalog of 12,346 non-isomorphic simple graphs on
> 8 vertices, the exact tested counting signature assigns a distinct signature to
> every catalog graph.

Because the signature was also checked under relabelings, this is a finite exhaustive
order-8 classification result for that exact signature and catalog.

It is **not** a claim for graphs with more than 8 vertices, not an asymptotic
graph-isomorphism result, and not a novelty or priority claim.
