# Matrix Calculus for Machine Learning and Beyond

This is the course page for an **18.063 Matrix Calculus** at MIT taught in **January 2025** ([IAP](https://elo.mit.edu/iap/)) by
Professors [Alan Edelman](https://math.mit.edu/~edelman/) and [Steven G. Johnson](https://math.mit.edu/~stevenj/).

* For past versions of this course, see [Matrix Calculus in IAP 2023 (OCW)](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/) on OpenCourseWare (also [on github](https://github.com/mitmath/matrixcalc/tree/iap2023), with videos [on YouTube](https://www.youtube.com/playlist?list=PLUl4u3cNGP62EaLLH92E_VCN4izBKK6OE)).  See also [Matrix Calculus in IAP 2022 (OCW)](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2022/pages/lecture-notes-and-readings/) (also [on github](https://github.com/mitmath/matrixcalc/tree/iap2022)), and [Matrix Calculus 2024 (github)](https://github.com/mitmath/matrixcalc/tree/iap2024); the previous years used the temporary 18.S096 "special subject" course number.

**Lectures:** MWF time 11am–1pm, Jan 13–Jan 31 in room 4-370; lecture recordings to be posted (MIT only).  3 units, *2 problem sets* due Jan 24 and Feb 31 — submitted electronically [via Canvas](https://canvas.mit.edu/courses/29776), no exams.  TA/grader: TBD.

**Course Notes**: [Draft notes from IAP 2024](https://www.dropbox.com/scl/fi/iq4plt8oqja845cuuosa4/Matrix-Calculus-latest.pdf?rlkey=nsnytdu28jje41nhh1bl2dbba&st=i6lfha0r&dl=0).  Other materials to be posted.

**Piazza forum:** Online discussions at [Piazza](https://piazza.com/mit/spring2025/18063).

**Description:**

> We all know that calculus courses such as 18.01 and 18.02 are univariate and vector calculus, respectively. Modern applications such as machine learning and large-scale optimization require the next big step, "matrix calculus" and calculus on arbitrary vector spaces.
>
> This class covers a coherent approach to matrix calculus showing techniques that allow you to think of a matrix holistically (not just as an array of scalars), generalize and compute derivatives of important matrix factorizations and many other complicated-looking operations, and understand how differentiation formulas must be re-imagined in large-scale computing. We will discuss reverse/adjoint/backpropagation differentiation, custom vector-Jacobian products, and how modern automatic differentiation is more computer science than calculus (it is neither symbolic formulas nor finite differences).

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

## Lecture 1 (Jan 13)

* part 1: overview ([slides](https://docs.google.com/presentation/d/16uwYARbg4unaGU4Enp6uQvlBb6N21j1UINQW99om6R4/edit?usp=sharing))
* part 2: derivatives as linear operators: matrix functions, gradients, product and chain rule
* [video (MIT only)](https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=87d1cb6c-c91a-4f4a-a573-b2630084197e)

 Re-thinking derivatives as linear operators: f(x+dx)-f(x)=df=f′(x)[dx] — f′ is the [linear operator](https://en.wikipedia.org/wiki/Linear_map) that gives the change df in the *output* from a "tiny" change dx in the *inputs*, to *first order* in dx (i.e. dropping higher-order terms).   When we have a vector function f(x)∈ℝᵐ of vector inputs x∈ℝⁿ, then f'(x) is a linear operator that takes n inputs to m outputs, which we can think of as an m×n matrix called the [Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) (typically covered only superficially in 18.02).

 In the same way, we can define derivatives of matrix-valued operators as linear operators on matrices.  For example, f(X)=X² gives f'(X)[dX] = X dX + dX X.  Or f(X) = X⁻¹ gives f'(X)[dX] = –X⁻¹ dX X⁻¹.   These are perfectly good linear operators acting on matrices dX, even though they are not written in the form (Jacobian matrix)×(column vector)!   (We *could* rewrite them in the latter form by reshaping the inputs dX and the outputs df into column vectors, more formally by choosing a basis, and we will later cover how this process can be made more elegant using [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product).  But for the most part it is neither necessary nor desirable to express all linear operators as Jacobian matrices in this way.)

 Reviewed the (easy) derivations of the sum rule d(f+g)=df+dg and the product rule d(fg) = (df)g+f(dg), directly from the definition of f(x+dx)-f(x)=df=f′(x)[dx], dropping higher-order terms.

 Discussed the chain rule for f(g(x)) (f'(x)=g'(h(x))h'(x), where this is a *composition* of two linear operations, performing h' then g' — g'h' ≠ h'g'!).   For functions from vectors to vectors, the chain rule is simply the *product of Jacobians*.   Moreover, as soon as you compose 3 or more functions, it can a make a huge difference whether you multiply the Jacobians from left-to-right ("reverse-mode", or "backpropagation", or "adjoint differentiation") or right-to-left ("forward-mode"). Showed, for example, that if you have *many inputs but a single output* (as is common in machine learning and other types of optimization problem), that it is vastly more efficient to multiply left-to-right than right-to-left, and such "backpropagation algorithms" are a key factor in the practicality of large-scale optimization.

**Further reading**: *Draft Course Notes* (link above), chapters 1 and 2.
 [matrixcalculus.org](http://www.matrixcalculus.org/) (linked in the slides) is a fun site to play with derivatives of matrix and vector functions.  The [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) has a lot of formulas for these derivatives, but no derivations.  Some [notes on vector and matrix differentiation](https://cdn-uploads.piazza.com/paste/j779e63owl53k6/04b2cb8c2f300212d723bea822a6b856085b28e28ca9debc75a05761a436499c/6.S087_Lecture_2.pdf) were posted for 6.S087 from IAP 2021.

 **Further reading (gradients)**: A fancy name for a row vector is a "covector" or [linear form](https://en.wikipedia.org/wiki/Linear_form), and the fancy version of the relationship between row and column vectors is the [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem), but until you get to non-Euclidean geometry you may be happier thinking of a row vector as the transpose of a column vector.

**Further reading (chain rule)**: The terms "forward-mode" and "reverse-mode" differentiation are most prevalent in [automatic differentiation (AD)](https://en.wikipedia.org/wiki/Automatic_differentiation), which will will cover later in this course. You can find many, many articles online about [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) in neural networks.   There are many other versions of this, e.g. in differential geometry the derivative linear operator (multiplying Jacobians and perturbations dx right-to-left) is called a [pushforward](https://en.wikipedia.org/wiki/Pushforward_(differential)), whereas multiplying a gradient row vector (covector) by a Jacobian left-to-right is called a [pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry)).   This [video on the principles of AD in Julia](https://www.youtube.com/watch?v=UqymrMG-Qi4) by [Dr. Mohamed Tarek](https://github.com/mohamed82008) also starts with a similar left-to-right (reverse) vs right-to-left (forward) viewpoint and goes into how it translates to Julia code, and how you define custom chain-rule steps for Julia AD.  In other fields, "reverse mode" is sometimes called an "adjoint method": see the [notes on adjoint methods](https://github.com/mitmath/18335/blob/spring21/notes/adjoint/adjoint.pdf) and [slides](https://github.com/mitmath/18335/blob/spring21/notes/adjoint/adjoint-intro.pdf) from 18.335 ([video](https://mit.zoom.us/rec/share/xLxMyhBSoIhSFxce5lHb1ubItby5BKFs6mgJJ7kMmjotETmaYm4YA22TA8w8n13i.6sTEFrkkloG7LFeR?startTime=1619463273000)).

**Further reading (fancier math)**: the perspective of derivatives as linear operators is sometimes called a [Fréchet derivative](https://en.wikipedia.org/wiki/Fr%C3%A9chet_derivative) and you can find lots of very abstract (what I'm calling "fancy") presentations of this online, chock full of weird terminology whose purpose is basically to generalize the concept to weird types of vector spaces.  The "little-o notation" o(δx) we're using here for "infinitesimal asymptotics" is closely related to the [asymptotic notation](https://en.wikipedia.org/wiki/Big_O_notation) used in computer science, but in computer science people are typically taking the limit as the argument (often called "n") becomes very *large* instead of very small.

## Lecture 2 (Jan 15)

* part 1: matrix Jacobians via [vectorization](https://en.wikipedia.org/wiki/Vectorization_(mathematics)); notes: [2×2 Matrix Jacobians (html)](https://rawcdn.githack.com/mitmath/matrixcalc/3f6758996e40c5c1070279f89f7f65e76e08003d/notes/2x2Jacobians.jl.html) [(pluto notebook source code)](https://github.com/mitmath/matrixcalc/blob/main/notes/2x2Jacobians.jl) [(jupyter notebook)](notes/2x2Jacobians.ipynb).  Course notes: chapter 3.

* example of reverse-mode/adjoint differentiation for optimizing g(p) = f(A(p)⁻¹b), showing why a linear-operator formula like d(A⁻¹)=–A⁻¹ dA A⁻¹ is actually perfectly practical and usable, and why evaluating the chain rule outputs-to-inputs is so practically important.  Last few slides of lecture 1.

* [video (MIT only)](https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=0abdc678-c7bd-4253-9e11-b263008420eb)


**Further reading**: Wikipedia has a useful list of [properties of the matrix trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Properties).  The "matrix dot product" introduced today is also called the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product), and the corresponding norm ("length" of the matrix viewed as a vector) is the [Frobenius norm](https://mathworld.wolfram.com/FrobeniusNorm.html).   When you "flatten" a matrix A by stacking its columns into a single vector, the result is called [vec(A)](https://en.wikipedia.org/wiki/Vectorization_(mathematics)), and many important linear operations on matrices can be expressed as [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product).  The [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) has lots of formulas for derivatives of matrix functions.  See the [notes on adjoint methods](https://github.com/mitmath/18335/blob/spring21/notes/adjoint/adjoint.pdf) and [slides](https://github.com/mitmath/18335/blob/spring21/notes/adjoint/adjoint-intro.pdf) from 18.335 ([video](https://mit.zoom.us/rec/share/xLxMyhBSoIhSFxce5lHb1ubItby5BKFs6mgJJ7kMmjotETmaYm4YA22TA8w8n13i.6sTEFrkkloG7LFeR?startTime=1619463273000)).

## Lecture 3 (Jan 17)

* generalized gradients and inner products — [handwritten notes](https://www.dropbox.com/s/dtdriu0jg1sqoqr/Derivatives%20as%20Linear%20Operators.pdf?dl=0)
 - also norms and derivatives: why a norm of the input and output are needed to *define* a derivative
 - more on handling units: when the components of the vector are quantities different units, defining the inner product (and hence the norm) requires dimensional weight factors to scale the quantities.  (Using standard gradient / inner product implicitly uses weights given by whatever units you are using.) A change of variables (to nondimensionalize the problem) is equivalent (for steepest descent) to a nondimensionalization of the inner-product/norm, but the former is typically easier for use with off-the-shelf optimization software.   Usually, you want to use units/scaling so that all your quantities have similar scales, otherwise steepest descent may converge very slowly!

* The [gradient of the determinant](https://rawcdn.githack.com/mitmath/matrixcalc/b08435612045b17745707f03900e4e4187a6f489/notes/determinant_and_inverse.html) is ∇(det A) = det(A)A⁻ᵀ

* an amazing trick by [Mathias (1996)](https://doi.org/10.1137/S0895479895283409): `f([A dA; 0I A]) = [f(A) f′(A)[dA]; 0I f(A)]` (in Julia notation) for any analytic/smooth function f(A) acting on square matrices.  (e.g. matrix powers/polynomials, matrix exponentials, etcetera).

* [video (MIT only)](https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=134a6f3a-b1b9-442e-8e53-b26300842887)

* pset 1: to be posted soon

Generalizing **gradients** to *scalar* functions f(x) for x in arbitrary *vector spaces* x ∈ V.   The key thing is that we need not just a vector space, but an **inner product** x⋅y (a "dot product", also denoted ⟨x,y⟩ or ⟨x|y⟩); V is then formally called a [Hilbert space](https://en.wikipedia.org/wiki/Hilbert_space).   Then, for *any* scalar function, since df=f'(x)[dx] is a linear operator mapping dx∈V to scalars df∈ℝ (a "[linear form](https://en.wikipedia.org/wiki/Linear_form)"), it turns out that it [*must* be a dot product](https://en.wikipedia.org/wiki/Riesz_representation_theorem) of dx with "something", and we call that "something" the gradient!  That is, once we define a dot product, then for any scalar function f(x) we can define ∇f by f'(x)[dx]=∇f⋅dx.  So ∇f is always something with the same "shape" as x (the [steepest-ascent](https://math.stackexchange.com/questions/223252/why-is-gradient-the-direction-of-steepest-ascent) direction).

Talked about the general [requirements for an inner product](https://en.wikipedia.org/wiki/Inner_product_space): linearity, positivity, and (conjugate) symmetry (and also mentioned the [Cauchy–Schwarz inequality](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality), which follows from these properties).  Gave some examples of inner products, such as the familiar Euclidean inner product xᵀy or a weighted inner product.  Defined the most obvious inner product of m×n matrices: the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product) A⋅B=`sum(A .* B)`=trace(AᵀB)=vec(A)ᵀvec(B), the sum of the products of the matrix entries.  This also gives us the "Frobenius norm" ‖A‖²=A⋅A=trace(AᵀA)=‖vec(A)‖², the square root of the sum of the squares of the entries.   Using this, we can now take the derivatives of various scalar functions of matrices, e.g. we considered

* f(A)=tr(A) ⥰ ∇f = I
* f(A)=‖A‖ ⥰ ∇f = A/‖A‖
* f(A)=xᵀAy ⥰ ∇f = xyᵀ (for constant x, y)
* f(A)=det(A) ⥰ ∇f = det(A)(A⁻¹)ᵀ = transpose of the [adjugate](https://en.wikipedia.org/wiki/Adjugate_matrix) of A

Also talked about the definition of a [norm](https://en.wikipedia.org/wiki/Norm_(mathematics)) (which can be obtained from an inner product if you have one, but can also be defined by itself), and why a norm is necessary to define a derivative: it is embedded in the definition of what a higher-order term o(δx) means.   (Although there are many possible norms, [in finite dimensions all norms are equivalent up to constant factors](https://math.mit.edu/~stevenj/18.335/norm-equivalence.pdf), so the definition of a derivative does not depend on the choice of norm.)

Made precise and derived (with the help of Cauchy–Schwarz) the well known fact that ∇f is the **steepest-ascent** direction, for *any* scalar-valued function on a vector space with an inner product (any Hilbert space), in the norm corresponding to that inner product.  That is, if you take a step δx with a fixed length ‖δx‖=s, the greatest increase in f(x) to first order is found in a direction parallel to ∇f.

Closed with a sketch of an amazing formula by Mathias for the derivatives of smooth functions from square matrices to square matrices, which you will investigate more for homework: for a sufficiently smooth function f(A) from square matrices to square matrices, it turns out that:
$$
f(\begin{bmatrix} A & \delta A \\\ & A \end{bmatrix}) =
    \begin{bmatrix} f(A) & f'(A)[\delta A] \\\ & f(A) \end{bmatrix} \, .
$$
(This is *exact* for any δA, even if it is not small!)

**Further reading (gradients and norms):** Course notes, chapter 5.

**Further reading (∇det)**: Course notes, chapter 7.  There are lots of discussions of the
[derivative of a determinant](https://en.wikipedia.org/wiki/Jacobi%27s_formula) online, involving the ["adjugate" matrix](https://en.wikipedia.org/wiki/Adjugate_matrix) det(A)A⁻¹.
Not as well documented is that the gradient of the determinant is the cofactor matrix widely used for the [Laplace expansion](https://en.wikipedia.org/wiki/Laplace_expansion) of a determinant.
The formula for the [derivative of log(det A)](https://statisticaloddsandends.wordpress.com/2018/05/24/derivative-of-log-det-x/) is also nice, and logs of determinants appear in surprisingly many applications (from statistics to quantum field theory).  The [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) contains many of these formulas, but no derivations.   A nice application of d(det(A)) is solving for eigenvalues λ by applying Newton's method to det(A-λI)=0, and more generally one can solve det(M(λ))=0 for any function Μ(λ) — the resulting roots λ are called [nonlinear eigenvalues](https://en.wikipedia.org/wiki/Nonlinear_eigenproblem) (if M is nonlinear in λ), and one can [apply Newton's method](https://www.maths.manchester.ac.uk/~ftisseur/talks/FT_talk2.pdf) using the determinant-derivative formula here.

**Further reading (Mathias formula):** The earliest derivation of this formula seems to be by [Mathias (1996)](https://doi.org/10.1137/S0895479895283409).  A generalization is proved in [Higham (2008)](https://epubs.siam.org/doi/book/10.1137/1.9780898717778): if $A$ is an $n \times n$ matrix, the function $f(A)$ need only be $2n-1$ times differentiable (or even only once differentiable if $A$ is [normal](https://en.wikipedia.org/wiki/Normal_matrix)), instead of requiring it to be [analytic](https://en.wikipedia.org/wiki/Analytic_function) (have a Taylor series).  The Mathias formula is very closely related to the power law and other rules for [2×2 Jordan blocks / generalized eigenvectors of defective matrices](https://web.mit.edu/18.06/www/Spring17/jordan-vectors.pdf), as well as to the [matrix representation of dual numbers](https://en.wikipedia.org/wiki/Dual_number#Matrix_representation).
