# V103 Preliminary Literature / Novelty Audit

**Snapshot date:** 2026-08-26

## Scope

This is a claim-safety and publication-readiness audit, not a proof of novelty.
It compares the frozen V101/V102 finite-n=10 results against directly relevant
graph reconstruction, polynomial/spectral reconstruction, and computational
complete-invariant literature found in the current search.

## What is clearly established prior art

1. **Full vertex-deck reconstruction at n=10 is not new.** McKay's exhaustive
   work establishes reconstruction for all graphs through n=13.
2. **Characteristic-polynomial / spectral information from vertex-deleted
   graphs is a classical research direction.** Polynomial reconstruction has
   been studied for roughly five decades and remains open in general for
   ordinary graphs.
3. **Combining multiple numerical graph invariants to obtain complete
   discrimination on n=9/n=10 is not new in the broad sense.** Dehmer,
   Emmert-Streib and Grabner (2014) exhaustively studied connected graphs with
   9 and 10 vertices and reported complete multivariate discrimination.
4. **Minimal supplementary information for reconstruction is also an existing
   research theme.** Sciriha–Borg study minimal parameters accompanying a
   labelled card.

## What appears genuinely more specific in Counting Revolution

The searched literature did **not** reveal a direct prior match for this exact
object:

> On the complete 12,005,168 non-isomorphic simple graphs of order 10, restrict
> each vertex-deleted card to the fixed aligned atomic family
> `{e1,...,e9,tree}`, and determine the exact minimum number of atomic fields as
> a function of an arbitrary auxiliary bit budget `b=0..6`.

The frozen exact frontier is:

`[5,3,3,2,2,2,1]`.

Likewise, no direct match was found in this search for:

- the exact 45-pair / 10-single obstruction atlas used to prove the bit-budget
  lower bounds;
- exact pair maximum-class results such as `e2+e6 -> 7`;
- the explicit finite-n=10 `e2+e4+e6 + 1 bit` construction using the V94
  mod-127 residue/coloring channel;
- the exact monotone Boolean-lattice minimum over the stated 10-field atomic
  family.

## Required novelty wording

Safe:

> "We report an exhaustive finite-n=10 computation for a fixed family of
> aligned deleted-card invariants and obtain the exact field-count/auxiliary-bit
> frontier [5,3,3,2,2,2,1]. In the literature searched for this audit, we did
> not identify a prior study reporting this same frontier."

Unsafe without a broader systematic review:

- "first ever"
- "world first"
- "new solution to graph isomorphism"
- "solution of the Reconstruction Conjecture"
- "first complete invariant for 10-vertex graphs"
- "first use of spectral/deleted-card invariants for reconstruction"
- "provably optimal graph invariant" without explicitly stating the fixed
  atomic family and finite n=10 scope.

## Most important close prior work

The 2014 Dehmer–Emmert-Streib–Grabner paper is the most important comparator
for any publication draft, because it already performs exhaustive n=9/n=10
complete discrimination using combinations of numerical graph invariants.
A paper must distinguish the present contribution by its *restricted
deleted-card atomic family*, *exact information-budget frontier*, *matching
lower/upper bounds*, and *explicit obstruction certificates*, rather than by
generic complete discrimination at n=10.

## References

1. Brendan D. McKay (2022). **Reconstruction of small graphs and digraphs**. Australasian Journal of Combinatorics 83(3), 448–457. https://ajc.maths.uq.edu.au/pdf/83/ajc_v83_p448.pdf
   - Relevance: Computer searches prove the graph reconstruction conjecture for graphs with up to 13 vertices. Establishes that ordinary full-deck reconstructibility at n=10 is not novel.
2. Brendan D. McKay (1997). **Small graphs are reconstructible**. Australasian Journal of Combinatorics 15, 123–126. https://users.cecs.anu.edu.au/~bdm/papers/recon.pdf
   - Relevance: Earlier exhaustive reconstruction work; full/isomorph-reduced decks distinguish small graphs.
3. J. A. Bondy and R. L. Hemminger (1977). **Graph reconstruction—a survey**. Journal of Graph Theory 1, 227–268. DOI: 10.1002/jgt.3190010306. https://onlinelibrary.wiley.com/doi/10.1002/jgt.3190010306
   - Relevance: Classical survey and context for Kelly–Ulam reconstruction.
4. W. T. Tutte (1979). **All the King's Horses (A Guide to Reconstruction)**. Graph Theory and Related Topics, Academic Press, pp. 15–33. https://books.google.com/books/about/Graph_Theory_and_Related_Topics.html?id=jujuAAAAMAAJ
   - Relevance: Classical reconstruction framework; characteristic-polynomial/determinantal reconstruction belongs to established theory.
5. Elias M. Hagos (2000). **The characteristic polynomial of a graph is reconstructible from the characteristic polynomials of its vertex-deleted subgraphs and their complements**. Electronic Journal of Combinatorics 7(1), R12. DOI: 10.37236/1490. https://www.combinatorics.org/ojs/index.php/eljc/article/view/v7i1r12
   - Relevance: Shows long-standing polynomial-deck reconstruction context and a positive result when complement polynomial deck is also supplied.
6. Irene Sciriha and Zoran Stanić (2023). **The polynomial reconstruction problem: The first 50 years**. Discrete Mathematics 346, 113349. DOI: 10.1016/j.disc.2023.113349. https://www.sciencedirect.com/science/article/abs/pii/S0012365X23000353
   - Relevance: Survey: polynomial reconstruction from the vertex-deleted characteristic-polynomial deck remains unresolved in general for ordinary graphs.
7. Alexander Farrugia (2021). **Graphs Having Most of Their Eigenvalues Shared by a Vertex Deleted Subgraph**. Symmetry 13(9), 1663. DOI: 10.3390/sym13091663. https://www.mdpi.com/2073-8994/13/9/1663
   - Relevance: Modern polynomial-deck reconstruction results; includes the derivative identity linking the parent characteristic polynomial to vertex-deleted polynomials.
8. Weifang Lv, Wei Wang, Wei Wang, Hao Zhang (2026). **The spectral reconstruction problem revisited**. Linear Algebra and its Applications 729, 1–23. DOI: 10.1016/j.laa.2025.09.020. https://www.sciencedirect.com/science/article/pii/S0024379525003933
   - Relevance: Current spectral-reconstruction literature; studies determination from graph spectrum plus vertex-deleted spectra.
9. Matthias Dehmer, Martin Grabner, Abbe Mowshowitz, Frank Emmert-Streib (2013). **An efficient heuristic approach to detecting graph isomorphism based on combinations of highly discriminating invariants**. Advances in Computational Mathematics 39(2), 311–325. DOI: 10.1007/s10444-012-9281-0. https://researchportal.tuni.fi/en/publications/an-efficient-heuristic-approach-to-detecting-graph-isomorphism-ba/
   - Relevance: Prior work explicitly combines discriminating graph invariants; broad 'combine invariants to distinguish graphs' novelty claims are unsafe.
10. Matthias Dehmer, Frank Emmert-Streib, Martin Grabner (2014). **A computational approach to construct a multivariate complete graph invariant**. Information Sciences 260, 200–208. DOI: 10.1016/j.ins.2013.11.008. https://www.sciencedirect.com/science/article/abs/pii/S0020025513007950
   - Relevance: Especially close computational precedent: exhaustive connected non-isomorphic graphs with 9 and 10 vertices; a 97-dimensional multivariate invariant and low-dimensional iterative invariant selection achieve perfect discrimination.
11. Irene Sciriha and James L. Borg (2024). **Reconstruction from one labelled card and more**. Linear Algebra and its Applications 693, 271–287. DOI: 10.1016/j.laa.2023.08.009. https://www.sciencedirect.com/science/article/pii/S0024379523003142
   - Relevance: Relevant minimal-information reconstruction theme: asks which graph parameters must accompany one card for unique reconstruction.
