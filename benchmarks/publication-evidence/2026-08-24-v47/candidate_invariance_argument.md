# Why `n_left_square_absorption` is an isomorphism invariant

For a magma `(S, *)`, define

`L = |{(x,y) in S^2 : x*x = x*y}|`.

Let `f : S -> T` be a magma isomorphism. For every ordered pair `(x,y)`,

`x*x = x*y`

holds if and only if, after applying `f` and using preservation of the operation,

`f(x) *' f(x) = f(x) *' f(y)`.

The map `(x,y) -> (f(x),f(y))` is a bijection from `S^2` to `T^2`.
Therefore it bijects the solution sets of the displayed identity, so their
cardinalities are equal.

This proves invariance under relabeling/isomorphism. It does not claim novelty.
