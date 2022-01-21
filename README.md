# Matrix Calculus for Machine Learning and Beyond

This is the course page for an 18.S096 Special Subject in Mathematics at MIT taught in **January 2022** ([IAP](https://elo.mit.edu/iap/)) by
Professors [Alan Edelman](https://math.mit.edu/~edelman/) and [Steven G. Johnson](https://math.mit.edu/~stevenj/).

**Lectures:** MWF 11am–1pm, Jan 10–28, *virtually* [via Zoom](https://mit.zoom.us/j/92847472121?pwd=K2xEZE1xUXVyY2xKZ0VyT2FMMTQrUT09).  3 units, *2 problem sets* due Jan 19 and Jan 26, no exams.  [Piazza discussions](https://piazza.com/mit/other/18s096). TA/grader: Gaurav Arya.

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

* part 1: [introductory slides](https://docs.google.com/presentation/d/1RqkL3AD6hVrUNpevQ7lhQ3InGr5-quQrnhQPsjvosDU/edit?usp=sharing)

* part 2: [derivatives as linear operators](https://www.dropbox.com/s/d7t8g9h19utqlcj/Fr%C3%A9chet%20Derivatives.pdf?dl=0) (handwritten notes)

* [lecture video](https://mit.zoom.us/rec/share/qi7oSsGUm6wKaCQ2foToEqgIieRymEQfX462F-mhHg1YZ49TiCm_p0jQHx4VE-Ll.x4C_yE7md1junk6M?startTime=1641830229000) (unfortunately missing a section of part 2).

**Further reading**: [matrixcalculus.org](http://www.matrixcalculus.org/) (linked in the slides) is a fun site to play with derivatives of matrix and vector functions.  The [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) has a lot of formulas for these derivatives, but no derivations.  Some [notes on vector and matrix differentiation](https://cdn-uploads.piazza.com/paste/j779e63owl53k6/04b2cb8c2f300212d723bea822a6b856085b28e28ca9debc75a05761a436499c/6.S087_Lecture_2.pdf) were posted for 6.S087 from IAP 2021.

**Further reading (fancier math)**: the perspective of derivatives as linear operators is sometimes called a [Fréchet derivative](https://en.wikipedia.org/wiki/Fr%C3%A9chet_derivative) and you can find lots of very abstract (what I'm calling "fancy") presentations of this online, chock full of weird terminology whose purpose is basically to generalize the concept to weird types of vector spaces.  The "little-o notation" o(δx) we're using here for "infinitesimal asymptotics" is closely related to the [asymptotic notation](https://en.wikipedia.org/wiki/Big_O_notation) used in computer science, but in computer science people are typically taking the limit as the argument (often called "n") becomes very *large* instead of very small.  A fancy name for a row vector is a "covector" or [linear form](https://en.wikipedia.org/wiki/Linear_form), and the fancy version of the relationship between row and column vectors is the [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem), but until you get to non-Euclidean geometry you may be happier thinking of a row vector as the transpose of a column vector.

## Lecture 2 (Jan 12)
* [video](https://mit.zoom.us/rec/share/QmAPatyA-0uih6FebzqgWb_i_6NUW-MG0vwUZwAKR46tkrXOWddBMqIThWoONEnU.EAVw3yrEBjRJYzft?startTime=1642003088000)
* part 1: [derivatives as linear operators](https://www.dropbox.com/s/d7t8g9h19utqlcj/Fr%C3%A9chet%20Derivatives.pdf?dl=0), continued from lecture 1.
* part 2: 2x2 Matrix Jacobians [(html)](https://rawcdn.githack.com/mitmath/matrixcalc/7340d2a7d40e6548a5ca0945ecae96cbac659929/2x2Jacobians.jl.html) [(pluto notebook source code)](https://github.com/mitmath/matrixcalc/blob/main/2x2Jacobians.jl)
* [pset 1](hw1.pdf) (due next Wed)

Continued discussing Jacobian matrices (for vector-valued functions of vectors), with some example.  Switched a streamlined "infinitesimal" notation df=f'(x)dx, where we now simply omit higher-order terms instead of writing o(δx), and f'(x) is taken to be a linear operator acting to the right on dx (≠ dx f'(x)!).  Sum, product, and chain rules for derivatives as linear operators.

The chain rule and associativity: forward vs. reverse differentiation.   The chain rule leads to a product of Jacobian matrices, and while you can't rearrange this product (matrix multiplication is *not commutative*) you can change the order in which you do the multiplications from left-to-right or right-to-left (matrix multiplication is *associative*).  It turns out that this ordering can have a huge impact on the practical speed at which you can compute derivatives in large problem.  Explained why, if you have 1 (or few) *outputs*, then you want to do Jacobian products from left-to-right so that you do vector–matrix products and not matrix–matrix products: this is called **reverse-mode** differentiation (also "adjoint" differentiation, or "backpropagation" in machine learning).  Conversely, if there is only 1 (or a few) *inputs*, then you want to do the chain rule from right-to-left, calledd **forward-mode** differentiation.   Gave an example of training neural networks, where there are zillions of inputs (the "fitting" parameters of the NN) but only one output (the "loss" function measuring the error compared to training data), and this leads to reverse-mode differentiation or "backpropagation".  (Unfortunately somewhat garbled during lecture, but written notes are cleaned up.)

In part 2 (last few minutes), began setting up some example problems involving matrix functions of matrices, and "[vectorization](https://en.wikipedia.org/wiki/Vectorization_(mathematics))" of matrices to vectors and linear operators to [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product).  To be continued.

**Further reading**: The terms "forward-mode" and "reverse-mode" differentiation are most prevalent in [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), which will will cover later in this course. You can find many, many articles online about [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) in neural networks.   There are many other versions of this, e.g. in differential geometry the derivative linear operator (multiplying Jacobians and perturbations dx right-to-left) is called a [pushforward](https://en.wikipedia.org/wiki/Pushforward_(differential)), whereas multiplying a gradient row vector (covector) by a Jacobian left-to-right is called a [pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry)).

## Lecture 3 (Jan 14)

* part 1: [slides](https://docs.google.com/presentation/d/1ov4Rl3wZ9ZbkYTDcCTHmyDiwHLYiGPYjCJjHKWWFiS4/edit?usp=sharing)
* part 1: Matrix Jacobians notebook from lecture 2 [(html)](https://rawcdn.githack.com/mitmath/matrixcalc/7340d2a7d40e6548a5ca0945ecae96cbac659929/2x2Jacobians.jl.html), [(pluto notebook source code)](https://github.com/mitmath/matrixcalc/blob/main/2x2Jacobians.jl)
* part 2: [notebook](https://nbviewer.org/github/mitmath/matrixcalc/blob/main/Finite%20difference%20checks.ipynb)
* [video](https://mit.zoom.us/rec/share/GUf5tZEboaxBSvSVID8wNaaudVFD2VeE3BBTYnlUASYL-vPRI621N2dPTIpFuJj7.jVbLIsHVyGK6wdd4?startTime=1642175965000)

Revisiting the gradient ∇f.   Scalar functions of matrices, matrix dot products, and the trace.  Matrix Jacobians, continued from lecture 2.   Finite-difference approximation.

**Further reading**: Wikipedia has a useful list of [properties of the matrix trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Properties).  The "matrix dot product" introduced today is also called the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product), and the corresponding norm ("length" of the matrix viewed as a vector) is the [Frobenius norm](https://mathworld.wolfram.com/FrobeniusNorm.html).   When you "flatten" a matrix A by stacking its columns into a single vector, the result is called [vec(A)](https://en.wikipedia.org/wiki/Vectorization_(mathematics)), and many important linear operations on matrices can be expressed as [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product).  The [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) has lots of formulas for derivatives of matrix functions.  There is a lot of information online on [finite difference approximations](https://en.wikipedia.org/wiki/Finite_difference),  [these 18.303 notes](https://github.com/mitmath/18303/blob/fall16/difference-approx.pdf), or [Section 5.7 of *Numerical Recipes*](http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf).   The Julia [FiniteDifferences.jl package](https://github.com/JuliaDiff/FiniteDifferences.jl) provides lots of algorithms to compute finite-difference approximations; a particularly robust and powerful way to obtain high accuracy is to employ [Richardson extrapolation](https://github.com/JuliaDiff/FiniteDifferences.jl#richardson-extrapolation) to smaller and smaller δx.  If you make δx too small, the finite precision (#digits) of [floating-point arithmetic](https://en.wikipedia.org/wiki/Floating-point_arithmetic) leads to [catastrophic cancellation errors](https://en.wikipedia.org/wiki/Catastrophic_cancellation).

## Lecture 4 (Jan 19)

* part 1: derivative of matrix determinant and inverse [(html)](https://rawcdn.githack.com/mitmath/matrixcalc/c97512521a9ff63802454ee258f1759c45f7d8b6/determinant_and_inverse.html) [(julia source)](determinant_and_inverse.jl)
* part 2: nonlinear root-finding, optimization, and adjoint-method differentiation [slides](https://docs.google.com/presentation/d/1U1lB5bhscjbxEuH5FcFwMl5xbHl0qIEkMf5rm0MO8uE/edit?usp=sharing)
* [video](https://mit.zoom.us/rec/share/Fp41z7ftWiSyn8oFR12pb4Xgyf5hEghj6paskaE_MUVPo5id-ZFRJeQqlqMRE-My.ZT8OEUhWsmpBc_U5?startTime=1642607958000) (missing lecture part 2, sorry!)
* [pset 1 solutions](hw1sol.pdf) and computational [notebook](https://nbviewer.org/github/mitmath/matrixcalc/blob/main/hw1sol.ipynb)
* [pset 2](hw2.pdf) (due Wed Jan 26)

**Further reading (part 1)**: There are lots of discussions of the
[derivative of a determinant](https://en.wikipedia.org/wiki/Jacobi%27s_formula) online, involving the ["adjugate" matrix](https://en.wikipedia.org/wiki/Adjugate_matrix) det(A)A⁻¹.
Not as well documented is that the gradient of the determinant is the cofactor matrix widely used for the [Laplace expansion](https://en.wikipedia.org/wiki/Laplace_expansion) of a determinant.
The formula for the [derivative of log(det A)](https://statisticaloddsandends.wordpress.com/2018/05/24/derivative-of-log-det-x/) is also nice, and logs of determinants appear in surprisingly many applications (from statistics to quantum field theory).  The [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) contains many of these formulas, but no derivations.   A nice application of d(det(A)) is solving for eigenvalues λ by applying Newton's method to det(A-λI)=0, and more generally one can solve det(M(λ))=0 for any function Μ(λ) — the resulting roots λ are called [nonlinear eigenvalues](https://en.wikipedia.org/wiki/Nonlinear_eigenproblem) (if M is nonlinear in λ), and one can [apply Newton's method](https://www.maths.manchester.ac.uk/~ftisseur/talks/FT_talk2.pdf) using the determinant-derivative formula here.

**Further reading (part 2)**:  SGJ gave another [overview of optimization](https://github.com/mitmath/18335/blob/spring21/notes/optimization.pdf) in 18.335 ([video](https://mit.zoom.us/rec/share/QwT0OMMFfkgi9dD0Zoa_3UK-14LbQR8GFcd7Q-O9PqIJTbbULGYqX3isDkLa1kOw.CNhZ0KukrqW2kxeT?startTime=1619031558000)).  There are many textbooks on [nonlinear optimization](http://www.athenasc.com/nonlinbook.html) algorithms of various sorts, including specialized books on [convex optimization](http://web.stanford.edu/~boyd/cvxbook/), [derivative-free optimization](http://bookstore.siam.org/mp08/), etcetera.  A useful review of topology-optimization methods can be found in [Sigmund and Maute (2013)](https://link.springer.com/article/10.1007/s00158-013-0978-6).  See the [notes on adjoint methods](https://github.com/mitmath/18335/blob/spring21/notes/adjoint/adjoint.pdf) and [slides](https://github.com/mitmath/18335/blob/spring21/notes/adjoint/adjoint-intro.pdf) from 18.335 ([video](https://mit.zoom.us/rec/share/xLxMyhBSoIhSFxce5lHb1ubItby5BKFs6mgJJ7kMmjotETmaYm4YA22TA8w8n13i.6sTEFrkkloG7LFeR?startTime=1619463273000)).

## Lecture 5 (Jan 21)

* [notes](https://rawcdn.githack.com/mitmath/matrixcalc/e90417f46a20bec6d9c743c6b7bf5b178e77913a/automatic_differentiation_done_quick.html)

Automatic differentiation, guest lecture by [Dr. Chris Rackauckas](https://chrisrackauckas.com/).
