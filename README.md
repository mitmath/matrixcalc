# Matrix Calculus for Machine Learning and Beyond

This is the course page for an **18.063 Matrix Calculus** at MIT taught in **January 2026** ([IAP](https://elo.mit.edu/iap/)) by
Professors [Alan Edelman](https://math.mit.edu/~edelman/) and [Steven G. Johnson](https://math.mit.edu/~stevenj/).

* For past versions of this course, see [Matrix Calculus in IAP 2023 (OCW)](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/) on OpenCourseWare (also [on github](https://github.com/mitmath/matrixcalc/tree/iap2023), with videos [on YouTube](https://www.youtube.com/playlist?list=PLUl4u3cNGP62EaLLH92E_VCN4izBKK6OE)).  See also [Matrix Calculus in IAP 2022 (OCW)](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2022/pages/lecture-notes-and-readings/) (also [on github](https://github.com/mitmath/matrixcalc/tree/iap2022)), and [Matrix Calculus 2024 (github)](https://github.com/mitmath/matrixcalc/tree/iap2024) and [Matrix Calculus 2025 (github)](https://github.com/mitmath/matrixcalc/tree/iap2025); some previous years used the temporary 18.S096 "special subject" course number.

**Lectures:** MWF time 11am–1pm, Jan 12–Jan 30 (except Jan 19) in room 35-310.  3 units, *2 problem sets* due Jan 23 and Jan 30 — submitted electronically [via Canvas](https://canvas.mit.edu/courses/35760), no exams.

**Course Notes**: [18.063 COURSE NOTES](https://www.dropbox.com/scl/fi/iq4plt8oqja845cuuosa4/Matrix-Calculus-latest.pdf?rlkey=nsnytdu28jje41nhh1bl2dbba&st=i6lfha0r&dl=0).  Other materials to be posted below.

**Piazza forum:** Online discussions at [Piazza](https://piazza.com/class/mkab8649oo96qm/).

**Description:**

> We all know that calculus courses such as 18.01 and 18.02 are univariate and vector calculus, respectively. Modern applications such as machine learning and large-scale optimization require the next big step, "matrix calculus" and calculus on arbitrary vector spaces.
>
> This class **revisits and generalizes calculus from the perspective of linear algebra**, extending it to much more general things (e.g. the derivative of matrix functions, like a matrix inverse or determinant with respect to the *matrix*, or an integral with respect to a *function*, an ODE solution with respect to ODE parameters) and connecting it to the computer science of efficient algorithms for differentiation and automatic differentiation (AD).
>
> We present a coherent approach to matrix calculus emphasizing matrices as holistic objects (not just as an array of scalars), we generalize and compute derivatives of important matrix factorizations and many other complicated-looking operations, and understand how differentiation formulas must be re-imagined in large-scale computing. We will discuss reverse/adjoint/backpropagation differentiation, custom vector-Jacobian products, and how modern AD is more computer science than calculus (it is neither symbolic formulas nor finite differences).

**Prerequisites:** Linear Algebra such as [18.06](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) and multivariate calculus such as [18.02](https://ocw.mit.edu/courses/mathematics/18-02-multivariable-calculus-fall-2007/).

Course will involve simple numerical computations using the [Julia language](https://github.com/mitmath/julia-mit).   Ideally install it on your own computer following [these instructions](https://github.com/mitmath/julia-mit#installing-julia-and-ijulia-on-your-own-computer), but as a fallback you can run it in the cloud here:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mitmath/binder-env/main)

**Topics:**

Here are some of the planned topics:

* Derivatives as linear operators and linear approximation on arbitrary vector spaces: beyond gradients and Jacobians.
* Derivatives of functions with matrix inputs and/or outputs (e.g. matrix inverses and determinants).  Kronecker products and matrix "vectorization".
* Derivatives of matrix factorizations (e.g. eigenvalues/SVD) and derivatives with constraints (e.g. orthogonal matrices).
* Multidimensional chain rules, and the significance of right-to-left ("forward") vs. left-to-right ("reverse") composition.  Chain rules on computational graphs (e.g. neural networks).
* Forward- and reverse-mode manual and automatic multivariate differentiation.
* Adjoint methods (vJp/pullback rules) for derivatives of solutions of linear, nonlinear, and differential equations.
* Application to nonlinear root-finding and optimization.  Multidimensional Newton and steepest–descent methods.
* Applications in engineering/scientific optimization and machine learning.
* Second derivatives, Hessian matrices, quadratic approximations, and quasi-Newton methods.

## Lecture 1 (Jan 12)

* part 1: overview ([slides](https://docs.google.com/presentation/d/16uwYARbg4unaGU4Enp6uQvlBb6N21j1UINQW99om6R4/edit?usp=sharing))
* part 2: derivatives as linear operators: matrix functions, gradients, product and chain rule

 Re-thinking derivatives as linear operators: f(x+dx)-f(x)=df=f′(x)[dx]. That is, f′ is the [linear operator](https://en.wikipedia.org/wiki/Linear_map) that gives the change df in the *output* from a "tiny" change dx in the *inputs*, to *first order* in dx (i.e. dropping higher-order terms).   When we have a vector function f(x)∈ℝᵐ of vector inputs x∈ℝⁿ, then f'(x) is a linear operator that takes n inputs to m outputs, which we can think of as an m×n matrix called the [Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) (typically covered only superficially in 18.02).

 In the same way, we can define derivatives of matrix-valued operators as linear operators on matrices.  For example, f(X)=X² gives f'(X)[dX] = X dX + dX X.  Or f(X) = X⁻¹ gives f'(X)[dX] = –X⁻¹ dX X⁻¹.   These are perfectly good linear operators acting on matrices dX, even though they are not written in the form (Jacobian matrix)×(column vector)!   (We *could* rewrite them in the latter form by reshaping the inputs dX and the outputs df into column vectors, more formally by choosing a basis, and we will later cover how this process can be made more elegant using [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product).  But for the most part it is neither necessary nor desirable to express all linear operators as Jacobian matrices in this way.)

**Further reading**: *Course Notes* (link above), chapters 1 and 2.
 [matrixcalculus.org](http://www.matrixcalculus.org/) (linked in the slides) is a fun site to play with derivatives of matrix and vector functions.  The [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) has a lot of formulas for these derivatives, but no derivations.  Some [notes on vector and matrix differentiation](https://cdn-uploads.piazza.com/paste/j779e63owl53k6/04b2cb8c2f300212d723bea822a6b856085b28e28ca9debc75a05761a436499c/6.S087_Lecture_2.pdf) were posted for 6.S087 from IAP 2021.

**Further reading (fancier math)**: the perspective of derivatives as linear operators is sometimes called a [Fréchet derivative](https://en.wikipedia.org/wiki/Fr%C3%A9chet_derivative) and you can find lots of very abstract (what I'm calling "fancy") presentations of this online, chock full of weird terminology whose purpose is basically to generalize the concept to weird types of vector spaces.  The "little-o notation" o(δx) we're using here for "infinitesimal asymptotics" is closely related to the [asymptotic notation](https://en.wikipedia.org/wiki/Big_O_notation) used in computer science, but in computer science people are typically taking the limit as the argument (often called "n") becomes very *large* instead of very small.  We will formalize this later, corresponding to **section 5.2** of the course notes.

## Lecture 2 (Jan 14)

* part 1: generalized sum and product rule, derivatives of X⁻¹ and ‖x‖² and xᵀAx; gradients ∇f of scalar-valued functions.  Blackboard + some [slides](https://docs.google.com/presentation/d/16uwYARbg4unaGU4Enp6uQvlBb6N21j1UINQW99om6R4/edit?usp=sharing) from lecture 1.  Course notes: **chapter 2**.
* part 1: matrix-function Jacobians via [vectorization](https://en.wikipedia.org/wiki/Vectorization_(mathematics)) and [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product); notes: [2×2 Matrix Jacobians (html)](https://rawcdn.githack.com/mitmath/matrixcalc/3f6758996e40c5c1070279f89f7f65e76e08003d/notes/2x2Jacobians.jl.html) [(pluto notebook source code)](https://github.com/mitmath/matrixcalc/blob/main/notes/2x2Jacobians.jl) [(jupyter notebook)](notes/2x2Jacobians.ipynb).  Course notes: **chapter 3**.

 **Further reading (gradients)**: We will cover more generalizations later, corresponding to **chapter 5** of the course notes. A fancy name for a row vector is a "covector" or [linear form](https://en.wikipedia.org/wiki/Linear_form), and the fancy version of the relationship between row and column vectors is the [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem), but until you get to non-Euclidean geometry you may be happier thinking of a row vector as the transpose of a column vector.

## Lecture 3 (Jan 16)

* part 1: the chain rule and forward vs. reverse "mode" differentiation: course notes **section 2.4**.  Example applications, **chapter 6**: slides on nonlinear root-finding, optimization, and adjoint-method differentiation [slides](https://docs.google.com/presentation/d/1U1lB5bhscjbxEuH5FcFwMl5xbHl0qIEkMf5rm0MO8uE/edit?usp=sharing)

* matrix gradients via the matrix inner product (the ["Frobenius" inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product))" course notes **chapter 5**

* [pset 1](psets/pset1.pdf) posted, due Friday Jan 23 at midnight.

## Lecture 4 (Jan 21)

* part 1: generalized gradients and inner products — [handwritten notes](https://www.dropbox.com/scl/fi/byg5mpcnnk4xh9tqjbjmk/Inner-Products-and-Norms.pdf?rlkey=egsdhyee9go9v17iuxxqx1edj&dl=0) and course notes **chapter 5**
    - also norms and derivatives: why a norm of the input and output are needed to *define* a derivative, and in particular to define what "higher-order terms" and o(δx) mean
    - more on handling units: when the components of the vector are quantities different units, defining the inner product (and hence the norm) requires dimensional weight factors to scale the quantities.  (Using standard gradient / inner product implicitly uses weights given by whatever units you are using.) A change of variables (to nondimensionalize the problem) is equivalent (for steepest descent) to a nondimensionalization of the inner-product/norm, but the former is typically easier for use with off-the-shelf optimization software.   Usually, you want to use units/scaling so that all your quantities have similar scales, otherwise steepest descent may converge very slowly!


* The [gradient of the determinant](https://rawcdn.githack.com/mitmath/matrixcalc/b08435612045b17745707f03900e4e4187a6f489/notes/determinant_and_inverse.html) is ∇(det A) = det(A)A⁻ᵀ (course notes **chapter 7**)

Generalizing **gradients** to *scalar* functions f(x) for x in arbitrary *vector spaces* x ∈ V.   The key thing is that we need not just a vector space, but an **inner product** x⋅y (a "dot product", also denoted ⟨x,y⟩ or ⟨x|y⟩); V is then formally called a [Hilbert space](https://en.wikipedia.org/wiki/Hilbert_space).   Then, for *any* scalar function, since df=f'(x)[dx] is a linear operator mapping dx∈V to scalars df∈ℝ (a "[linear form](https://en.wikipedia.org/wiki/Linear_form)"), it turns out that it [*must* be a dot product](https://en.wikipedia.org/wiki/Riesz_representation_theorem) of dx with "something", and we call that "something" the gradient!  That is, once we define a dot product, then for any scalar function f(x) we can define ∇f by f'(x)[dx]=∇f⋅dx.  So ∇f is always something with the same "shape" as x (the [steepest-ascent](https://math.stackexchange.com/questions/223252/why-is-gradient-the-direction-of-steepest-ascent) direction).

Talked about the general [requirements for an inner product](https://en.wikipedia.org/wiki/Inner_product_space): linearity, positivity, and (conjugate) symmetry (and also mentioned the [Cauchy–Schwarz inequality](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality), which follows from these properties).  Gave some examples of inner products, such as the familiar Euclidean inner product xᵀy or a weighted inner product.  Defined the most obvious inner product of m×n matrices: the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product) A⋅B=`sum(A .* B)`=trace(AᵀB)=vec(A)ᵀvec(B), the sum of the products of the matrix entries.  This also gives us the "Frobenius norm" ‖A‖²=A⋅A=trace(AᵀA)=‖vec(A)‖², the square root of the sum of the squares of the entries.   Using this, we can now take the derivatives of various scalar functions of matrices, e.g. we considered

* f(A)=tr(A) ⥰ ∇f = I
* f(A)=‖A‖ ⥰ ∇f = A/‖A‖
* f(A)=xᵀAy ⥰ ∇f = xyᵀ (for constant x, y)
* f(A)=det(A) ⥰ ∇f = det(A)(A⁻¹)ᵀ = transpose of the [adjugate](https://en.wikipedia.org/wiki/Adjugate_matrix) of A

Also talked about the definition of a [norm](https://en.wikipedia.org/wiki/Norm_(mathematics)) (which can be obtained from an inner product if you have one, but can also be defined by itself), and why a norm is necessary to define a derivative: it is embedded in the definition of what a higher-order term o(δx) means.   (Although there are many possible norms, [in finite dimensions all norms are equivalent up to constant factors](https://math.mit.edu/~stevenj/18.335/norm-equivalence.pdf), so the definition of a derivative does not depend on the choice of norm.)

Made precise and derived (with the help of Cauchy–Schwarz) the well known fact that ∇f is the **steepest-ascent** direction, for *any* scalar-valued function on a vector space with an inner product (any Hilbert space), in the norm corresponding to that inner product.  That is, if you take a step δx with a fixed length ‖δx‖=s, the greatest increase in f(x) to first order is found in a direction parallel to ∇f.

**Further reading (∇det)**: Course notes, chapter 7.  There are lots of discussions of the
[derivative of a determinant](https://en.wikipedia.org/wiki/Jacobi%27s_formula) online, involving the ["adjugate" matrix](https://en.wikipedia.org/wiki/Adjugate_matrix) det(A)A⁻¹.
Not as well documented is that the gradient of the determinant is the cofactor matrix widely used for the [Laplace expansion](https://en.wikipedia.org/wiki/Laplace_expansion) of a determinant.
The formula for the [derivative of log(det A)](https://statisticaloddsandends.wordpress.com/2018/05/24/derivative-of-log-det-x/) is also nice, and logs of determinants appear in surprisingly many applications (from statistics to quantum field theory).  The [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) contains many of these formulas, but no derivations.   A nice application of d(det(A)) is solving for eigenvalues λ by applying Newton's method to det(A-λI)=0, and more generally one can solve det(M(λ))=0 for any function Μ(λ) — the resulting roots λ are called [nonlinear eigenvalues](https://en.wikipedia.org/wiki/Nonlinear_eigenproblem) (if M is nonlinear in λ), and one can [apply Newton's method](https://www.maths.manchester.ac.uk/~ftisseur/talks/FT_talk2.pdf) using the determinant-derivative formula here.

## Lecture 5 (Jan 23)
* Directional derivatives: $f'(x)[v] = \frac{d}{d\alpha} f(x + \alpha v) \left. \right|_{\alpha=0}$.  Connection to "components" of gradient or derivative = directional derivative when $v$ is a Cartesian basis vector.  course notes **section 2.2.1**
* Reverse-mode gradients for neural networks (NNs): [handwritten backpropagation notes](https://www.dropbox.com/scl/fi/bke4pbr342e1jhv9qytg1/NN-Backpropagation.pdf?rlkey=b7krtzdt4hgsj63zyq9ok2gqv&dl=0), course notes **chapter 9**.
* forward-mode automatic differentiation (AD) via [dual numbers](https://en.wikipedia.org/wiki/Dual_number) ([Julia notebook](notes/AutoDiff.ipynb)) - course notes, **chapter 8**
* [pset 1 solutions](psets/pset1sol.pdf)
* [pset 2](psets/pset2.pdf): due midnight Jan 30

**Further reading on backpropagation for NNs**:  [Strang (2019)](https://math.mit.edu/~gs/learningfromdata/) section VII.3 and [18.065 OCW lecture 27](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-27-backpropagation-find-partial-derivatives/). You can find many, many articles online about [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) in neural networks. Backpropagation for neural networks is closely related to [backpropagation/adjoint methods for recurrence relations (course notes)](https://math.mit.edu/~stevenj/18.336/recurrence2.pdf), and on [computational graphs (blog post)](https://colah.github.io/posts/2015-08-Backprop/).  We will return to computational graphs in a future lecture.

**Further reading on forward AD**: Course notes, chapter 8.  Googling "automatic differentiation" will turn up many, many resources — this is a huge practical field these days.   [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) (described detail by [this paper](https://arxiv.org/abs/1607.07892)) in Julia uses [dual number](https://en.wikipedia.org/wiki/Dual_number) arithmetic similar to lecture to compute derivatives; see also this [AMS review article](http://www.ams.org/publicoutreach/feature-column/fc-2017-12), or google "dual number automatic differentiation" for many other reviews.    Adrian Hill posted some nice [lecture notes on automatic differentiation (Julia-based)](https://adrhill.github.io/julia-ml-course/L6_Automatic_Differentiation/) for an ML course at TU Berlin (Summer 2023).  [TaylorDiff.jl](https://github.com/JuliaDiff/TaylorDiff.jl) extends this to higher-order derivatives.

## Lecture 6 (Jan 26): via [Zoom (MIT only)](https://mit.zoom.us/j/98915152715?pwd=4GftZplphHYx7QIlDUL4vgiwzD7Rxc.1)

Due to the snow emergency, Monday's lecture will be held via Zoom at the link above.
