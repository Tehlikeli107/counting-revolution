# Counting Revolution V101 — finite n=10 information-frontier consolidation

This package freezes the exact atomic-field / auxiliary-bit frontier for
budgets 0 through 6.

| auxiliary bits | exact minimum atomic fields | witness |
|---:|---:|---|
| 0 | 5 | V75 five-field complete subsets |
| 1 | 3 | V94 `e2+e4+e6+h` |
| 2 | 3 | V92 `e2+e4+tree`, max class 4 |
| 3 | 2 | V100 `e2+e6`, max class 7 |
| 4 | 2 | V91 `e2+e4`, max class 10 |
| 5 | 2 | V91 `e2+e4`, max class 10 |
| 6 | 1 | V90/V90.1 `e4`, max class 57 |

Thus the exact vector is:

`[5,3,3,2,2,2,1]`

for auxiliary budgets `b=0,...,6`.

The important thresholds are:

- first budget allowing 3 fields: **1 bit**
- first budget allowing 2 fields: **3 bits**
- first budget allowing 1 field: **6 bits**

The result is exact only for the complete finite `n=10` graph catalog and
the explicit deleted-card atomic family `{e1,...,e9,tree}`.
