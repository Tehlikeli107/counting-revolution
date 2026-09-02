# Independent raw frontier replication (`v0.2.0`)

This directory contains the independent V119 raw-replication source used to
reproduce the finite order-10 information frontier

`[5,3,3,2,2,2,1]`.

The pipeline does **not** use the historical V73 result.

## Input

The script locates or downloads Brendan McKay's official `graph10.g6.gz` and
requires:

- compressed bytes: `31,112,164`
- compressed SHA-256:
  `a16f47a95e3e174f4b08042fec95dce8b67712b0e465b5097ffd9334dde2faf8`
- decompressed bytes: `120,051,680`
- decompressed SHA-256:
  `923cabf28082cba3ee296251d23eee21b32056b36cf4952e42958d468357df36`

## Run

Recommended Python: 3.11.

`python -m pip install -r requirements-v119.txt`

`python COUNTING_REVOLUTION_V119_INDEPENDENT_RAW_FRONTIER_REPLICATION.py`

The `.v119_work` directory is resumable.

## Completed reference run

- atomic dataset bytes: `4,802,067,200`
- atomic dataset SHA-256:
  `6ca20a0b6deb6a931fefd4b777b268e2d254d6a7a3eed118d574913976d9d5b9`
- zero-bit complete counts (sizes 1..5): `[0,0,0,0,5]`
- direct zero-bit incomplete-subset counterexamples: `632`
- direct pair/single obstruction non-isomorphism checks: `5,730`
- explicit one-bit residual classes separated: `125/125`
- final frontier: `[5,3,3,2,2,2,1]`
- historical V73 result used: `false`

The compact result ZIP is distributed in the `v0.2.0` release asset rather
than committed to Git history.

Reference result ZIP SHA-256:

`4b0d4d6b50529bbf24eca45b5f26039a88890da83b467044ba8c1221ac562e10`

## Scope

Finite `n=10`, fixed aligned atomic family only. No general graph-isomorphism,
Reconstruction Conjecture, asymptotic, `n>10`, or novelty/priority claim.
