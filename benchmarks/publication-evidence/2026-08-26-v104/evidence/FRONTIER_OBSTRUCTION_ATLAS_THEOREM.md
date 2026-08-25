# Counting Revolution V102 — frontier obstruction atlas

V101 freezes the exact finite-n=10 field-count frontier:

`[5,3,3,2,2,2,1]` for auxiliary budgets `0..6`.

V102 turns the main information-theoretic lower bounds into compact explicit
graph witnesses.

## Atomic pairs

For every one of the 45 atomic pairs, V102 stores **five pairwise
non-isomorphic graphs with exactly the same aligned two-field deleted-card
signature**.

Therefore every pair has a collision class of size at least 5.

This directly rules out:

- pair + arbitrary 1 bit (2 states)
- pair + arbitrary 2 bits (4 states)

by the pigeonhole principle.

## Single atomic fields

For every one of the 10 atomic fields, V102 stores **33 pairwise
non-isomorphic graphs with the same single-field deleted-card signature**.

Therefore every single field has a collision class of size at least 33.

This directly rules out:

- single field + arbitrary 3 bits (8 states)
- single field + arbitrary 4 bits (16 states)
- single field + arbitrary 5 bits (32 states)

V90/V90.1 give the stronger global single-field result that the unique best
single field `e4` still requires 57 states.

## Scope

All statements are exact only for the complete finite `n=10` catalog and the
explicit atomic family `{e1,...,e9,tree}`.
