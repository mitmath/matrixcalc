# Matrix Calculus for Machine Learning and Beyond

This is the course page for an 18.S096 Special Subject in Mathematics at MIT taught in **January 2022** ([IAP](https://elo.mit.edu/iap/)) by
Professors [Alan Edelman](https://math.mit.edu/~edelman/) and [Steven G. Johnson](https://math.mit.edu/~stevenj/).

**Lectures:** MWF 11am–1pm, Jan 10–28, *virtually* [via Zoom](https://mit.zoom.us/j/92847472121?pwd=K2xEZE1xUXVyY2xKZ0VyT2FMMTQrUT09).  3 units, *2 problem sets* due Jan 19 and Jan 26, no exams.  TA/grader: Gaurav Arya.

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

* part 1: [derivatives as linear operators](https://www.dropbox.com/s/d7t8g9h19utqlcj/Fr%C3%A9chet%20Derivatives.pdf?dl=0), continued from lecture 1.

* part 2: [2x2 matrix jacobians (download pdf)](https://github.com/mitmath/matrixcalc/raw/main/%F0%9F%8E%88%202x2Jacobians.jl%20%E2%80%94%20Pluto.jl.pdf)
*         [2x2 matrix jacobians (ugly raw html but you can download)](https://github.com/mitmath/matrixcalc/raw/main/2x2Jacobians%20(static%20html).html)


More to come.
