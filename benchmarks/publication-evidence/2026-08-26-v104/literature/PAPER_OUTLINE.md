# Proposed paper structure

## Working title
**Exact information frontiers for aligned vertex-deleted graph invariants at order ten**

Avoid "Counting Revolution" as the scientific title until novelty positioning is
settled; it can remain the repository/project name.

## 1. Introduction
- Reconstruction / graph-invariant context.
- Distinguish full deck, polynomial deck, and the restricted atomic card summary.
- State finite n=10 scope immediately.
- Cite McKay 1997/2022, Sciriha–Stanić 2023, Dehmer et al. 2013/2014.

## 2. Atomic deleted-card model
- Complete n=10 catalog identity: 12,005,168 non-isomorphic simple graphs.
- Per deleted card: e1,...,e9 plus exact spanning-tree count.
- Preserve within-card field alignment.
- Define field subset projection and auxiliary b-bit channel.

## 3. Exact information frontier
State the main computational theorem:
`m(b) = [5,3,3,2,2,2,1]` for `b=0,...,6`.

For every b:
- lower-bound witness/source,
- matching upper-bound construction,
- exact class-size argument.

## 4. Explicit obstruction atlas
- 45 pair bases × five non-isomorphic collisions.
- 10 single bases × 33 non-isomorphic collisions.
- Pigeonhole lower bounds for arbitrary side channels.

## 5. Constructive upper bounds
- zero-bit five-field witnesses,
- V94 one-bit explicit construction,
- e2+e4+tree max 4,
- e2+e6 max 7,
- e2+e4 max 10,
- e4 max 57.

## 6. Reproducibility
- official graph10 identity,
- exact manifest/SHA chain,
- independent direct-card checks,
- no hash equality in scientific partitions,
- Windows/Python/NumPy/Numba environment.

## 7. Relation to prior work
- Full deck reconstruction at this order already known.
- Broad complete multivariate invariants at n=10 already exist.
- Polynomial/spectral deck is classical.
- Present contribution is the exact frontier for the stated restricted atomic family.

## 8. Limitations
- finite n=10 only;
- chosen atomic family only;
- class-rank upper channels can be catalog-relative;
- V94 mask is finite-catalog calibrated;
- no general GI or reconstruction-conjecture claim;
- novelty still needs broader systematic review.

## 9. Discussion / next theory questions
- Does the frontier stabilize or exhibit a law for n=11?
- Can V94's bit be replaced by a natural mask-free invariant?
- Can obstruction families be lifted into infinite constructions?
- Which frontier entries admit theory rather than exhaustive computation?
