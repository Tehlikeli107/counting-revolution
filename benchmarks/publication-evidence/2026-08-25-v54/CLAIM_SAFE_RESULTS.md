# Counting Revolution V54 — order-8 top-level component minimality

The V51 full signature has **13 top-level components**.

V54 exhaustively checks all:

- 13 single-component subsets;
- 78 two-component subsets;
- 286 three-component subsets.

## Result

- Complete singles: **0**
- Complete pairs: **0**
- Complete triples: **13**
- Minimum cardinality within these 13 top-level components: **3**

The best two-component partition reaches **12,345/12,346** and
leaves one two-graph collision. The shared collision is:

- catalog index 266: `G?AFBo`
- catalog index 348: `G?B@`w`

The only top-level partition equivalence found is:

`traces_A1_to_An ≡ characteristic_coefficients`

which is expected because each determines the characteristic polynomial information
encoded by the other for this fixed order.

There are **13** exact complete three-component witnesses; see
`complete_triples.csv`.

## Claim boundary

Supported:

> On the committed V51 table for McKay's 12,346 order-8 non-isomorphic simple
> graphs, no one- or two-component subset of the 13 top-level signature components
> is collision-free, while 13 three-component subsets are collision-free.
> Therefore the minimum cardinality is 3 within this explicit 13-component space.

Not supported:

- global minimality over subfeatures or all possible graph invariants;
- completeness for n>8;
- asymptotic or complexity claims;
- novelty or priority.
