# Matrix Calculus for Machine Learning and Beyond

This is the course page for an 18.S096 Special Subject in Mathematics at MIT taught in **January 2022** ([IAP](https://elo.mit.edu/iap/)) by
Professors [Alan Edelman](https://math.mit.edu/~edelman/) and [Steven G. Johnson](https://math.mit.edu/~stevenj/).

**Lectures:** MWF 11am–1pm, Jan 10–28, *virtually* [via Zoom](https://mit.zoom.us/j/92847472121?pwd=K2xEZE1xUXVyY2xKZ0VyT2FMMTQrUT09).  3 units, *2 problem sets* due Jan 19 and Jan 26, no exams.

**Description:**

> We all know that calculus courses such as 18.01 and 18.02 are univariate and vector calculus, respectively. Modern applications such as machine learning require the next big step, matrix calculus.
>
> This class covers a coherent approach to matrix calculus showing techniques that allow you to think of a matrix holistically (not just as an array of scalars), compute derivatives of important matrix factorizations, and really understand forward and reverse modes of differentiation. We will discuss adjoint methods, custom Jacobian matrix vector products, and how modern automatic differentiation is more computer science than mathematics in that it is neither symbolic nor finite differences.

**Prerequisites:** Linear Algebra such as [18.06](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) and multivariate calculus such as [18.02](https://ocw.mit.edu/courses/mathematics/18-02-multivariable-calculus-fall-2007/).

Course will involve simple numerical compuations using the [Julia language](https://github.com/mitmath/julia-mit).   Ideally install it on your own computer following [these instructions](https://github.com/mitmath/julia-mit#installing-julia-and-ijulia-on-your-own-computer), but as a fallback you can run it in the cloud here: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mitmath/binder-env/main)

**Topics:**

The following is a list of topics we intend to cover, and roughly the order in which we will cover them.  Subject to revision as the course progresses.

* Multidimensional Taylor expansions, derivatives as linear operators, and multidimensional chain rule.
* Forward- and reverse-mode differentiation for manual and automatic multivariate differentiation.  Chain rules on computational graphs (e.g. neural networks).
* Derivatives of matrix functions and factorizations.
* Adjoint methods and vJp/pullback rules.
* Applications in engineering/scientific optimization, machine learning.
* Exterior calculus.   Linear operators and Kronecker products.

## Lecture 1 (Jan 10)

Materials and notes for lecture 1 will be posted here.
