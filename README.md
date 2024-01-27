# Matrix Calculus for Machine Learning and Beyond

This is the course page for an 18.S096 Special Subject in Mathematics at MIT taught in **January 2024** ([IAP](https://elo.mit.edu/iap/)) by
Professors [Alan Edelman](https://math.mit.edu/~edelman/) and [Steven G. Johnson](https://math.mit.edu/~stevenj/).

* For the previous version of this course, see [Matrix Calculus in IAP 2023 (OCW)](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/) on OpenCourseWare (also [on github](https://github.com/mitmath/matrixcalc/tree/iap2023), with videos [on YouTube](https://www.youtube.com/playlist?list=PLUl4u3cNGP62EaLLH92E_VCN4izBKK6OE)).  See also [Matrix Calculus in IAP 2022 (OCW)](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2022/pages/lecture-notes-and-readings/) (also [on github](https://github.com/mitmath/matrixcalc/tree/iap2022)).

**Lectures:** MWF time 11am–1pm, ~~Jan 16~~ Jan 17–Feb 2 in room ~~2-131~~ 2-105; lecture recordings posted on [Panopto (MIT only)](https://mit.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx?folderID=0862c67f-5052-48c5-a5a3-b0f5015cf498).  3 units, *2 problem sets* due Jan 26 and Feb 2 — submitted electronically [via Canvas](https://canvas.mit.edu/courses/24125), no exams.  TA/grader: Fedir Yudin.

**Course Notes**: [Draft notes from IAP 2023](https://www.dropbox.com/scl/fi/wmo5g2lfb599hwsulnees/Matrix-Calculus.pdf?rlkey=n9xll4xlref9n6lrr3zsmiv12&dl=0).  Other materials to be posted.

**Piazza forum:** Online discussions at [this piazza link](https://piazza.com/mit/spring2024/18s096).

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

## Lecture 1 (Jan 17)

* part 1: overview ([slides](https://docs.google.com/presentation/d/14TDndt-dWTYhjncv6Kms-opTx7KJbJF0KI0Le2VdazQ/edit?usp=sharing))
* part 2: derivatives as linear operators
* [video (MIT only)](https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=04ba7bb3-569d-4b17-a764-b0f70083e47b)

 Re-thinking derivatives as linear operators: f(x+dx)-f(x)=df=f′(x)[dx] — f′ is the [linear operator](https://en.wikipedia.org/wiki/Linear_map) that gives the change df in the *output* from a "tiny" change dx in the *inputs*, to *first order* in dx (i.e. dropping higher-order terms).   When we have a vector function f(x)∈ℝᵐ of vector inputs x∈ℝⁿ, then f'(x) is a linear operator that takes n inputs to m outputs, which we can think of as an m×n matrix called the [Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) (typically covered only superficially in 18.02).

 In the same way, we can define derivatives of matrix-valued operators as linear operators on matrices.  For example, f(X)=X² gives f'(X)[dX] = X dX + dX X.  Or f(X) = X⁻¹ gives f'(X)[dX] = –X⁻¹ dX X⁻¹.   These are perfectly good linear operators acting on matrices dX, even though they are not written in the form (Jacobian matrix)×(column vector)!   (We *could* rewrite them in the latter form by reshaping the inputs dX and the outputs df into column vectors, more formally by choosing a basis, and we will later cover how this process can be made more elegant using [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product).  But for the most part it is neither necessary nor desirable to express all linear operators as Jacobian matrices in this way.)

 Reviewed the (easy) derivations of the sum rule d(f+g)=df+dg and the product rule d(fg) = (df)g+f(dg), directly from the definition of f(x+dx)-f(x)=df=f′(x)[dx], dropping higher-order terms.

**Further reading**: *Draft Course Notes* (link above), sections 1, 2.1, 2.2, 2.4, 3.1, 3.3.
 [matrixcalculus.org](http://www.matrixcalculus.org/) (linked in the slides) is a fun site to play with derivatives of matrix and vector functions.  The [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) has a lot of formulas for these derivatives, but no derivations.  Some [notes on vector and matrix differentiation](https://cdn-uploads.piazza.com/paste/j779e63owl53k6/04b2cb8c2f300212d723bea822a6b856085b28e28ca9debc75a05761a436499c/6.S087_Lecture_2.pdf) were posted for 6.S087 from IAP 2021.

**Further reading (fancier math)**: the perspective of derivatives as linear operators is sometimes called a [Fréchet derivative](https://en.wikipedia.org/wiki/Fr%C3%A9chet_derivative) and you can find lots of very abstract (what I'm calling "fancy") presentations of this online, chock full of weird terminology whose purpose is basically to generalize the concept to weird types of vector spaces.  The "little-o notation" o(δx) we're using here for "infinitesimal asymptotics" is closely related to the [asymptotic notation](https://en.wikipedia.org/wiki/Big_O_notation) used in computer science, but in computer science people are typically taking the limit as the argument (often called "n") becomes very *large* instead of very small.


## Lecture 2 (Jan 19)

* part 1: more examples, gradients of scalar-valued functions, the chain rule, and forward/reverse mode differentiation (*slides* from lecture 1).  Course notes: sections 2.3, 3.2, 7.1, 8.3.
* part 2: matrix Jacobians via [vectorization](https://en.wikipedia.org/wiki/Vectorization_(mathematics)); notes: [2×2 Matrix Jacobians (html)](https://rawcdn.githack.com/mitmath/matrixcalc/3f6758996e40c5c1070279f89f7f65e76e08003d/notes/2x2Jacobians.jl.html) [(pluto notebook source code)](https://github.com/mitmath/matrixcalc/blob/main/notes/2x2Jacobians.jl).  Course notes: sections 4, 5.
* [pset 1](psets/pset1.pdf), due Friday, Jan 26.
* [video (MIT only)](https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=a2ae74c3-87c2-45f1-ab4e-b0f70083ec17)

Began with a review and a discussion of different viewpoints of linear operators.  Although linear operators (in finite dimensions) can always be *represented* with matrices, we don't *have* to use a matrix representation, and a common confusion (or notational shorthand) is to conflate this representation of the operator with the operator itself.

Worked out some more examples of derivatives, from the slides.

When we have a scalar function f(x)∈ℝ of vector inputs x∈ℝⁿ, then this gives us a "row vector" f′(x) since f′(x)dx is a scalar, which we interpret as the *transpose* of the gradient ∇f (which we call a "column" vector), i.e. **df = (∇f)⋅dx = (∇f)ᵀdx**.   We can generalize gradients to *scalar* functions f(x) for x in *arbitrary vector spaces* x ∈ V.   The key thing is that we need not just a vector space, but an **inner product** x⋅y (a "dot product", also denoted ⟨x,y⟩ or ⟨x|y⟩); V is then formally called a [Hilbert space](https://en.wikipedia.org/wiki/Hilbert_space).   Then, for *any* scalar function, since df=f'(x)[dx] is a linear operator mapping dx∈V to scalars df∈ℝ (a "[linear form](https://en.wikipedia.org/wiki/Linear_form)"), it turns out that it *must* be a dot product of dx with "something", and we call that "something" the gradient!  That is, once we define a dot product, then for any scalar function f(x) we can define ∇f by f'(x)[dx]=∇f⋅dx.  So ∇f is always something with the same "shape" as x (the [steepest-ascent](https://math.stackexchange.com/questions/223252/why-is-gradient-the-direction-of-steepest-ascent) direction).

Discussed the chain rule for f(g(x)) (f'(x)=g'(h(x))h'(x), where this is a *composition* of two linear operations, performing h' then g' — g'h' ≠ h'g'!).   For functions from vectors to vectors, the chain rule is simply the *product of Jacobians*.   Moreover, as soon as you compose 3 or more functions, it can a make a huge difference whether you multiply the Jacobians from left-to-right ("reverse-mode", or "backpropagation", or "adjoint differentiation") or right-to-left ("forward-mode"). Showed, for example, that if you have *many inputs but a single output* (as is common in machine learning and other types of optimization problem), that it is vastly more efficient to multiply left-to-right than right-to-left, and such "backpropagation algorithms" are a key factor in the practicality of large-scale optimization.  Gave an example (see slides) of a scalar-valued function f(A(p)⁻¹b), in which left-to-right order allows us to compute ∇f using only two linear solves: one with A and a second "adjoint" solve involving Aᵀ.  (We will talk more about such "adjoint methods" later.)

Part 2: Another viewpoint on derivatives of matrix-valued functions via their relationship to the "Jacobian matrix" picture.  Converting f′(A) to a conventional "Jacobian matrix" in such cases requires converting matrices A into column vectors vec(A), a process called "vectorization" of the matrix (by a common convention: simply stacking the matrix by columns).  Linear operators like f′(A)[dA]=AdA+dAA can then be expressed as "ordinary" matrices via [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product).

**Further reading (gradients)**: A fancy name for a row vector is a "covector" or [linear form](https://en.wikipedia.org/wiki/Linear_form), and the fancy version of the relationship between row and column vectors is the [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem), but until you get to non-Euclidean geometry you may be happier thinking of a row vector as the transpose of a column vector.

**Further reading (chain rule)**: The terms "forward-mode" and "reverse-mode" differentiation are most prevalent in [automatic differentiation (AD)](https://en.wikipedia.org/wiki/Automatic_differentiation), which will will cover later in this course. You can find many, many articles online about [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) in neural networks.   There are many other versions of this, e.g. in differential geometry the derivative linear operator (multiplying Jacobians and perturbations dx right-to-left) is called a [pushforward](https://en.wikipedia.org/wiki/Pushforward_(differential)), whereas multiplying a gradient row vector (covector) by a Jacobian left-to-right is called a [pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry)).   This [video on the principles of AD in Julia](https://www.youtube.com/watch?v=UqymrMG-Qi4) by [Dr. Mohamed Tarek](https://github.com/mohamed82008) also starts with a similar left-to-right (reverse) vs right-to-left (forward) viewpoint and goes into how it translates to Julia code, and how you define custom chain-rule steps for Julia AD.  In other fields, "reverse mode" is sometimes called an "adjoint method": see the [notes on adjoint methods](https://github.com/mitmath/18335/blob/spring21/notes/adjoint/adjoint.pdf) and [slides](https://github.com/mitmath/18335/blob/spring21/notes/adjoint/adjoint-intro.pdf) from 18.335 ([video](https://mit.zoom.us/rec/share/xLxMyhBSoIhSFxce5lHb1ubItby5BKFs6mgJJ7kMmjotETmaYm4YA22TA8w8n13i.6sTEFrkkloG7LFeR?startTime=1619463273000)).

**Further reading (vectorization)**:  When you "flatten" a matrix A by stacking its columns into a single vector, the result is called [vec(A)](https://en.wikipedia.org/wiki/Vectorization_(mathematics)), and many important linear operations on matrices can be expressed as [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product).


## Lecture 3 (Jan 22)

* part 1: forward-mode automatic differentiation via dual numbers ([notes](notes/AutoDiff.ipynb))
* part 2: finite-differences ([notes](notes/Finite%20difference%20checks.ipynb))
* [video (MIT only)](https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=6121f8ca-3226-4b8a-943d-b0fe0083f434)

**Further reading (part 1)**: Course notes, section 10.1.  Googling "automatic differentiation" will turn up many, many resources — this is a huge practical field these days.   [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) (described detail by [this paper](https://arxiv.org/abs/1607.07892)) in Julia uses [dual number](https://en.wikipedia.org/wiki/Dual_number) arithmetic similar to lecture to compute derivatives; see also this [AMS review article](http://www.ams.org/publicoutreach/feature-column/fc-2017-12), or google "dual number automatic differentiation" for many other reviews.    Adrian Hill posted some nice [lecture notes on automatic differentiation (Julia-based)](https://adrhill.github.io/julia-ml-course/L6_Automatic_Differentiation/) for an ML course at TU Berlin (Summer 2023).  [TaylorDiff.jl](https://github.com/JuliaDiff/TaylorDiff.jl) extends this to higher-order derivatives.

**Further reading (part 2)**: Course notes, section 6.  There is a lot of information online on [finite difference approximations](https://en.wikipedia.org/wiki/Finite_difference),  [these 18.303 notes](https://github.com/mitmath/18303/blob/fall16/difference-approx.pdf), or [Section 5.7 of *Numerical Recipes*](http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf).   The Julia [FiniteDifferences.jl package](https://github.com/JuliaDiff/FiniteDifferences.jl) provides lots of algorithms to compute finite-difference approximations; a particularly robust and powerful way to obtain high accuracy is to employ [Richardson extrapolation](https://github.com/JuliaDiff/FiniteDifferences.jl#richardson-extrapolation) to smaller and smaller δx.  If you make δx too small, the finite precision (#digits) of [floating-point arithmetic](https://en.wikipedia.org/wiki/Floating-point_arithmetic) leads to [catastrophic cancellation errors](https://en.wikipedia.org/wiki/Catastrophic_cancellation).


## *Optional* Julia Tutorial: Monday Jan 22 @ 5pm via Zoom

* Virtually via Zoom.  [Recording (MIT only)](https://mit.zoom.us/rec/share/oirQFHELtxJkopybssFzml7YrudRyvIlmXjmgq4YemqmjT0P7wMGCd9ilC7SMZ_o.iBZ-UO6_ww9WjwF0?startTime=1705960822000).

A basic overview of the Julia programming environment for numerical computations that we will use in 18.06 for simple computational exploration.   This (Zoom-based) tutorial will cover what Julia is and the basics of interaction, scalar/vector/matrix arithmetic, and plotting — we'll be using it as just a "fancy calculator" and no "real programming" will be required.

* [Tutorial materials](https://github.com/mitmath/julia-mit) (and links to other resources)

If possible, try to install Julia on your laptop beforehand using the instructions at the above link.  Failing that, you can run Julia in the cloud (see instructions above).


## Lecture 4 (Jan 24)

*  [Handwritten notes](https://www.dropbox.com/scl/fi/byg5mpcnnk4xh9tqjbjmk/Inner-Products-and-Norms.pdf?rlkey=egsdhyee9go9v17iuxxqx1edj&dl=0)
* generalized gradients and inner products — see handwritten notes
* norms and derivatives: why norms of the input and output vector spaces are needed to *define* a derivative - see handwritten notes
* steepest-ascent: why the gradient is the steepest-ascent direction for any inner product, using the corresponding norm.  example application: steepest-ascent optimization when x components have different units requires a weighted inner product and a corresponding norm - see handwritten notes
* slides on nonlinear root-finding, optimization, and adjoint-method differentiation [slides](https://docs.google.com/presentation/d/1U1lB5bhscjbxEuH5FcFwMl5xbHl0qIEkMf5rm0MO8uE/edit?usp=sharing) - talked about optimization only
* [video (MIT only)](https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=0547884e-2a94-4696-acdb-b0fe0083f36d)

Generalizing **gradients** to *scalar* functions f(x) for x in arbitrary *vector spaces* x ∈ V.   The key thing is that we need not just a vector space, but an **inner product** x⋅y (a "dot product", also denoted ⟨x,y⟩ or ⟨x|y⟩); V is then formally called a [Hilbert space](https://en.wikipedia.org/wiki/Hilbert_space).   Then, for *any* scalar function, since df=f'(x)[dx] is a linear operator mapping dx∈V to scalars df∈ℝ (a "[linear form](https://en.wikipedia.org/wiki/Linear_form)"), it turns out that it [*must* be a dot product](https://en.wikipedia.org/wiki/Riesz_representation_theorem) of dx with "something", and we call that "something" the gradient!  That is, once we define a dot product, then for any scalar function f(x) we can define ∇f by f'(x)[dx]=∇f⋅dx.  So ∇f is always something with the same "shape" as x (the [steepest-ascent](https://math.stackexchange.com/questions/223252/why-is-gradient-the-direction-of-steepest-ascent) direction).

Talked about the general [requirements for an inner product](https://en.wikipedia.org/wiki/Inner_product_space): linearity, positivity, and (conjugate) symmetry (and also mentioned the [Cauchy–Schwarz inequality](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality), which follows from these properties).  Gave some examples of inner products, such as the familiar Euclidean inner product xᵀy or a weighted inner product.  Defined the most obvious inner product of m×n matrices: the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product) A⋅B=`sum(A .* B)`=trace(AᵀB)=vec(A)ᵀvec(B), the sum of the products of the matrix entries.  This also gives us the "Frobenius norm" ‖A‖²=A⋅A=trace(AᵀA)=‖vec(A)‖², the square root of the sum of the squares of the entries.   Using this, we can now take the derivatives of various scalar functions of matrices, e.g. we considered

* f(A)=‖A‖ ⥰ ∇f = A/‖A‖
* f(A)=xᵀAy ⥰ ∇f = xyᵀ (for constant x, y)
* f(A)=det(A) ⥰ ∇f = det(A)(A⁻¹)ᵀ = [adjugate](https://en.wikipedia.org/wiki/Adjugate_matrix)(A)ᵀ: we will prove this later

Also talked about the definition of a [norm](https://en.wikipedia.org/wiki/Norm_(mathematics)) (which can be obtained from an inner product if you have one, but can also be defined by itself), and why a norm is necessary to define a derivative: it is embedded in the definition of what a higher-order term o(δx) means.   (Although there are many possible norms, [in finite dimensions all norms are equivalent up to constant factors](https://math.mit.edu/~stevenj/18.335/norm-equivalence.pdf), so the definition of a derivative does not depend on the choice of norm.)

Made precise and derived (with the help of Cauchy–Schwarz) the well known fact that ∇f is the **steepest-ascent** direction, for *any* scalar-valued function on a vector space with an inner product (any Hilbert space), in the norm corresponding to that inner product.  That is, if you take a step δx with a fixed length ‖δx‖=s, the greatest increase in f(x) to first order is found in a direction parallel to ∇f.

**Further reading:** Course notes, section 7 and section 8.2.   SGJ gave another [overview of optimization](https://github.com/mitmath/18335/blob/spring21/notes/optimization.pdf) in 18.335 ([video](https://mit.zoom.us/rec/share/QwT0OMMFfkgi9dD0Zoa_3UK-14LbQR8GFcd7Q-O9PqIJTbbULGYqX3isDkLa1kOw.CNhZ0KukrqW2kxeT?startTime=1619031558000)).  There are many textbooks on [nonlinear optimization](http://www.athenasc.com/nonlinbook.html) algorithms of various sorts, including specialized books on [convex optimization](http://web.stanford.edu/~boyd/cvxbook/), [derivative-free optimization](http://bookstore.siam.org/mp08/), etcetera.

## Lecture 5 (Jan 24)

* More on handling units - a change of variables (to nondimensionalize the problem) is equivalent (for steepest descent) to a nondimensionalization of the inner-product/norm, but the former is typically easier for use with off-the-shelf optimization software.   See end of [handwritten notes](https://www.dropbox.com/scl/fi/byg5mpcnnk4xh9tqjbjmk/Inner-Products-and-Norms.pdf?rlkey=egsdhyee9go9v17iuxxqx1edj&dl=0) from last lecture.
* Adjoint/reverse-mode gradients for linear and nonlinear equations - see [slides](https://docs.google.com/presentation/d/1U1lB5bhscjbxEuH5FcFwMl5xbHl0qIEkMf5rm0MO8uE/edit?usp=sharing) on nonlinear root-finding, optimization, and adjoint-method differentiation.  Example from [IAP 2023 pset 2](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/resources/mit18_s096iap23_pset2sol_pdf/).
* Reverse-mode gradients for neural networks: [handwritten backpropagation notes](https://www.dropbox.com/scl/fi/bke4pbr342e1jhv9qytg1/NN-Backpropagation.pdf?rlkey=b7krtzdt4hgsj63zyq9ok2gqv&dl=0)
* The [gradient of the determinant](https://rawcdn.githack.com/mitmath/matrixcalc/b08435612045b17745707f03900e4e4187a6f489/notes/determinant_and_inverse.html) is ∇(det A) = det(A)A⁻ᵀ
* pset 1 solutions - coming soon
* [pset 2](psets/pset2.pdf) (due midnight Feb 2)
