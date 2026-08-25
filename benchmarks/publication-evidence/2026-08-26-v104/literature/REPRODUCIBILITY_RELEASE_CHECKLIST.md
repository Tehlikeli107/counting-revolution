# Reproducibility / release checklist

## Evidence freeze
- [x] V101 frontier consolidation exists.
- [x] V102 obstruction atlas exists.
- [x] Source ZIP byte/SHA provenance is explicit.
- [x] Complete graph10 catalog identity is frozen upstream.
- [ ] Create one clean release directory containing only publication-relevant
      scripts/results, not the entire historical V-series.
- [ ] Re-run the release bundle from a clean environment.
- [ ] Record wall-clock/runtime and peak-memory measurements for the key scans.
- [ ] Add a machine-readable theorem dependency DAG.
- [ ] Add a one-command verification path that does not repeat all 12M scans
      when certificate verification is sufficient.
- [ ] Add a separate optional "from raw graph10 catalog" replication path.

## Mathematical presentation
- [x] Fixed atomic family explicitly defined.
- [x] Card-alignment semantics explicitly defined.
- [x] Lower vs upper proofs distinguished.
- [x] V90.1 rank-proof correction incorporated.
- [ ] Rewrite all historical terminology into one stable notation.
- [ ] Give exact proofs of fast identities used for e2/e3/e5/e6/etc.
- [ ] Provide a compact proof of auxiliary-state optimum = maximum base-class
      size when the full V83 globally injective key is ranked class-relatively.

## Claim safety
- [x] No general GI claim.
- [x] No Reconstruction Conjecture claim.
- [x] No n>10 extrapolation.
- [x] No broad "first complete n=10 invariant" claim.
- [ ] Complete a broader systematic novelty search before "novel/first" wording.
- [ ] Compare directly against Dehmer et al. 2014 in the manuscript.
