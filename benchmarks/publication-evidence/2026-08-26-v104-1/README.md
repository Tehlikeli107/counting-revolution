# V104.1 dependency-DAG integrity correction

This is an **additive metadata correction** for the publication evidence
released as `v0.1.0`.

## What was wrong

The frozen V104 file:

`benchmarks/publication-evidence/2026-08-26-v104/reproducibility/THEOREM_DEPENDENCY_DAG.json`

used four edge endpoints that were not defined in its `nodes` object:

- `V86`
- `V88`
- `V97.1`
- `V104`

The original release verifier checked file hashes, certificates, frontier,
obstruction-atlas counts and novelty wording, but did not validate dependency
graph endpoints or acyclicity.

## What this correction changes

Only the dependency graph metadata is corrected:

- define `V86`, `V88`, `V97.1`, and `V104`;
- preserve the original 22 directed edges;
- preserve the exact frontier `[5,3,3,2,2,2,1]`;
- add a verifier for:
  - every edge endpoint is defined;
  - no self-loop;
  - no duplicate edge;
  - graph is acyclic;
  - every node can reach the release root `V104`;
  - required correction nodes exist;
  - root result is unchanged.

## What this correction does **not** change

- no graph-catalog scan is rerun;
- no V101/V102/V103 certificate is changed;
- no theorem value changes;
- no obstruction atlas changes;
- no claim/novelty scope changes;
- the published `v0.1.0` tag and release remain immutable historical evidence.

Applied additive repository path:

`benchmarks/publication-evidence/2026-08-26-v104-1/`

## Publication status

The correction is applied additively on `master`.

The `v0.1.1` tag records the initial correction checkpoint but its GitHub
Release was intentionally left unpublished after a stale preview-status field
was detected in this correction metadata. The finalized correction metadata is
released from the subsequent `v0.1.2` checkpoint.

No scientific theorem, certificate payload, obstruction atlas, or `v0.1.0`
historical release is changed by this metadata finalization.
