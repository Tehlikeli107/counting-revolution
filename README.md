# Counting Revolution: Counting-Based Algebraic and Graph Invariants

> **Research status:** computational exploration of counting-valued invariants. The repository contains finite exhaustive experiment scripts and promising observations, but it does **not** establish a general graph-isomorphism algorithm, super-exponential scaling theorem, or externally validated state-of-the-art benchmark.

The central idea is simple: instead of recording only whether a law or pattern is satisfied, record **how often** it is satisfied. This produces richer finite signatures for algebraic structures and graphs.

## Experiment family 1: binary operations on small sets

For a magma `(S, *)`, examples of count-valued features include:

- number of associative triples;
- number of commuting pairs;
- identity/zero/idempotency counts;
- alternative/flexibility counts;
- selected sorted maps or multisets derived from the operation table.

`complete_classifier.py` enumerates all `3^(3^2) = 19,683` binary operations on a 3-element set, canonicalizes them under all relabelings, and searches for a small invariant set that separates the resulting isomorphism classes.

This is a finite exhaustive computation when rerun successfully. The script is designed to write `theorem_results/classifier_results.json`, but that result directory/artifact is not currently committed in the repository. Therefore the historical “3,330/3,330 complete classification” headline should be treated as a **script-reported finite result pending reproducible artifact capture**.

### Scaling claim

Historical versions of this README described the amplification from Boolean to counting signatures as **super-exponential** based on experiments at very small set sizes. That asymptotic claim is not established by the current evidence. Finite increases at `|S|=2,3,4` can motivate a scaling hypothesis, but they do not prove an asymptotic growth law.

## Experiment family 2: graph signatures

The graph scripts combine several isomorphism invariants, including examples such as:

- degree sequences;
- closed-walk traces / characteristic-polynomial information;
- component and distance statistics;
- spanning-tree counts;
- local neighborhood statistics;
- induced 4-vertex pattern summaries.

`graph_n8_exhaustive.py` is designed to download Brendan McKay's catalog of all 12,346 non-isomorphic graphs on 8 vertices, compute the combined signature for each graph, and check for collisions.

If a complete catalog is loaded and the script returns 12,346 distinct signatures with zero collisions, that is a valid **finite computational verification for that exact signature on n=8**. It is not a theorem that the same signature is complete for arbitrary graph order.

The repository currently does not commit a provenance-bound raw run artifact containing the catalog checksum, per-graph signatures/collisions, environment, and independently recomputed summary. The README therefore does not promote the historical “exhaustive proof” wording to an externally verified result.

## Relation to Weisfeiler–Leman

A specific pair can witness that one invariant family distinguishes graphs that 1-WL does not. The Shrikhande graph and the 4×4 rook graph are a standard difficult pair for color refinement, and the repository explores additional counts on such examples.

A successful separation of one or more WL-hard pairs is a **finite witness**, not a general statement that the full counting signature is strictly stronger than WL on every graph family or every dimension.

## Relation to graph reconstruction

Subgraph counts are deeply connected to graph-isomorphism and reconstruction theory. However, the general Graph Reconstruction Conjecture remains open: even whether the multiset of all `(n-1)`-vertex-deleted subgraphs determines every finite simple graph is not settled in full generality.

Accordingly, this repository makes no claim that a small fixed collection of polynomial-time counts solves graph isomorphism or reconstruction in general.

## GPU benchmark status

`gpu_fingerprinter.py` explores batch GPU calculation of graph features. Historical README versions reported approximately `2.88M graphs/sec` for an n=8 batch.

That number is **not currently supported by a committed raw timing artifact** with exact hardware, software versions, warmup protocol, repetitions, raw timings, and correctness output. It should be treated as a historical local measurement rather than a reproducible benchmark claim.

## What is currently defensible

- counting-valued laws can refine Boolean classifications on finite enumerated structures;
- small algebraic structures can be exhaustively canonicalized and tested against candidate signatures;
- combined graph invariants can be collision-tested against complete finite graph catalogs;
- selected induced-subgraph counts can distinguish graph pairs that simpler invariants may merge;
- these experiments motivate further study of compact, interpretable structural signatures.

## What is not established here

- a new general graph-isomorphism solution;
- universal completeness of the listed graph signature;
- super-exponential asymptotic amplification;
- general strict superiority over Weisfeiler–Leman;
- publication-grade GPU throughput comparisons;
- priority/novelty claims for counting law-satisfaction values without a dedicated literature review.

## Reproducibility priorities

For the finite results to become strong public evidence, capture:

1. exact source commit;
2. input catalog/data checksums;
3. environment/package versions;
4. raw per-instance signatures or sufficient collision logs;
5. independently recomputed class/collision counts;
6. raw benchmark timings;
7. explicit finite scope in every theorem/result statement.

## Installation

```bash
pip install numpy networkx
```

PyTorch/CUDA is optional for GPU experiments.

## License status

No repository-level `LICENSE` file is currently committed. Until provenance and licensing are explicitly resolved, do not infer reuse rights from earlier README wording.
