# Matrix Calculus for Machine Learning and Beyond

This is the course page for an 18.S096 Special Subject in Mathematics at MIT taught in **January 2023** ([IAP](https://elo.mit.edu/iap/)) by
Professors [Alan Edelman](https://math.mit.edu/~edelman/) and [Steven G. Johnson](https://math.mit.edu/~stevenj/).

For a previous version of this course, see [Matrix Calculus in IAP 2022 (OCW)](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2022/pages/lecture-notes-and-readings/) (also [on github](https://github.com/mitmath/matrixcalc/tree/iap2022)).

**Lectures:** MWF 11am–1pm, Jan 18–Feb 3 in room **2-190** (+ [video recording (MIT only)](https://mit.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx?folderID=a72e3378-1c0f-425d-ae52-af880157e85c) and posted notes).  3 units, *2 problem sets* due Jan 25 and Feb 1 — submitted electronically [via Canvas](https://canvas.mit.edu/courses/17880), no exams.  Piazza discussion forum TBA. TA/grader: TBA.

**Piazza forum:** ask questions on the [18.S096 piazza](https://piazza.com/mit/spring2023/18s096)

**Description:**

> We all know that calculus courses such as 18.01 and 18.02 are univariate and vector calculus, respectively. Modern applications such as machine learning and large-scale optimization require the next big step, "matrix calculus" and calculus on arbitrary vector spaces.
>
> This class covers a coherent approach to matrix calculus showing techniques that allow you to think of a matrix holistically (not just as an array of scalars), generalize and compute derivatives of important matrix factorizations and many other complicated-looking operations, and understand how differentiation formulas must be re-imagined in large-scale computing. We will discuss reverse/adjoint/backpropagation differentiation, custom vector-Jacobian products, and how modern automatic differentiation is more computer science than calculus (it is neither symbolic formulas nor finite differences).

**Prerequisites:** Linear Algebra such as [18.06](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) and multivariate calculus such as [18.02](https://ocw.mit.edu/courses/mathematics/18-02-multivariable-calculus-fall-2007/).

Course will involve simple numerical compuations using the [Julia language](https://github.com/mitmath/julia-mit).   Ideally install it on your own computer following [these instructions](https://github.com/mitmath/julia-mit#installing-julia-and-ijulia-on-your-own-computer), but as a fallback you can run it in the cloud here:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mitmath/binder-env/main)

**Topics:**

Here are some of the planned topics:

* Derivatives as linear operators and linear approximation on arbitrary vector spaces: beyond gradients and Jacobians.
* Derivatives of functions with matrix inputs and/or outputs (e.g. matrix inverses and determinants).  Kronecker products and matrix "vectorization".
* Derivatives of matrix factorizations (e.g. eigenvalues/SVD) and derivatives with constraints (e.g. orthogonal matrices).
* Multidimensional chain rules, and the signifance of right-to-left ("forward") vs. left-to-right ("reverse") composition.  Chain rules on computational graphs (e.g. neural networks).
* Forward- and reverse-mode manual and automatic multivariate differentiation.
* Adjoint methods (vJp/pullback rules) for derivatives of solutions of linear, nonlinear, and differential equations.
* Application to nonlinear root-finding and optimization.  Multidimensional Newton and steepest–descent methods.
* Applications in engineering/scientific optimization and machine learning.
* Second derivatives, Hessian matrices, quadratic approximations, and quasi-Newton methods.

## Lecture 1 (Jan 18)

* part 1: [introductory slides](https://docs.google.com/presentation/d/1RqkL3AD6hVrUNpevQ7lhQ3InGr5-quQrnhQPsjvosDU/edit?usp=sharing)
* part 2: derivatives as linear operators — [handwritten notes](https://www.dropbox.com/s/dtdriu0jg1sqoqr/Derivatives%20as%20Linear%20Operators.pdf?dl=0)
* video recording (MIT only): [zoom recording](https://mit.zoom.us/rec/share/zsxeWYPY-zUW63ENT000dYdTt-vvVqvKYfM7J2J1QAvPuHMAXvadv4FDrtXsh_M.1JmilTjYrXV9ycwD?startTime=1674057910000) and [classroom recording](https://mit.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx?folderID=a72e3378-1c0f-425d-ae52-af880157e85c)

Part 1: Overview, applications, and motivation.

Part 2: Re-thinking derivatives as linear operators: f(x+dx)-f(x)=df=f′(x)[dx] — f′ is the [linear operator](https://en.wikipedia.org/wiki/Linear_map) that gives the change df in the *output* from a "tiny" change dx in the *inputs*, to *first order* in dx (i.e. dropping higher-order terms).   When we have a scalar function f(x)∈ℝ of vector inputs x∈ℝⁿ, then this gives us a "row vector" f′(x) since f′(x)dx is a scalar, which we interpret as the *transpose* of the gradient ∇f (which we call a "column" vector), i.e. **df = (∇f)⋅dx = (∇f)ᵀdx**.   When we have a vector function f(x)∈ℝᵐ of vector inputs x∈ℝⁿ, then f'(x) is a linear operator that takes n inputs to m outputs, which we can think of as an m×n matrix called the [Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) (typically covered only superficially in 18.02.)

**Further reading**: [matrixcalculus.org](http://www.matrixcalculus.org/) (linked in the slides) is a fun site to play with derivatives of matrix and vector functions.  The [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) has a lot of formulas for these derivatives, but no derivations.  Some [notes on vector and matrix differentiation](https://cdn-uploads.piazza.com/paste/j779e63owl53k6/04b2cb8c2f300212d723bea822a6b856085b28e28ca9debc75a05761a436499c/6.S087_Lecture_2.pdf) were posted for 6.S087 from IAP 2021.

**Further reading (fancier math)**: the perspective of derivatives as linear operators is sometimes called a [Fréchet derivative](https://en.wikipedia.org/wiki/Fr%C3%A9chet_derivative) and you can find lots of very abstract (what I'm calling "fancy") presentations of this online, chock full of weird terminology whose purpose is basically to generalize the concept to weird types of vector spaces.  The "little-o notation" o(δx) we're using here for "infinitesimal asymptotics" is closely related to the [asymptotic notation](https://en.wikipedia.org/wiki/Big_O_notation) used in computer science, but in computer science people are typically taking the limit as the argument (often called "n") becomes very *large* instead of very small.  A fancy name for a row vector is a "covector" or [linear form](https://en.wikipedia.org/wiki/Linear_form), and the fancy version of the relationship between row and column vectors is the [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem), but until you get to non-Euclidean geometry you may be happier thinking of a row vector as the transpose of a column vector.

## Lecture 2 (Jan 20)
* part 0: examples of linear and nonlinear transformations of ℝ² via images — [try it online](https://mit-c25.netlify.app/notebooks/1_hyperbolic_corgi)
* part 1: derivatives as linear operators — [handwritten notes](https://www.dropbox.com/s/dtdriu0jg1sqoqr/Derivatives%20as%20Linear%20Operators.pdf?dl=0)
* video recording (MIT only): [zoom recording](https://mit.zoom.us/rec/share/Ib9qBKqPFD85-IykgmyEVrpQB7ijB1bcebQMBPTarLWR3U7eqSvH13Hem81qBpKM.oHFTKkoUubxoji-B?startTime=1674230719000) and [classroom recording](https://mit.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx?folderID=a72e3378-1c0f-425d-ae52-af880157e85c)
* part 2: matrix Jacobians via [vectorization](https://en.wikipedia.org/wiki/Vectorization_(mathematics)); notes: [2×2 Matrix Jacobians (html)](https://rawcdn.githack.com/mitmath/matrixcalc/3f6758996e40c5c1070279f89f7f65e76e08003d/notes/2x2Jacobians.jl.html) [(pluto notebook source code)](https://github.com/mitmath/matrixcalc/blob/main/notes/2x2Jacobians.jl)
* [pset 1](psets/pset1.pdf) due Jan 25

Part 1: Continued discussing derivatives as linear operators, starting with Jacobian matrices.  Reviewed the sum rule d(f+g)=df+dg, the product rule d(fg) = (df)g+f(dg), and the chain rule for f(g(x)) (f'(x)=g'(h(x))h'(x), where this is a *composition* of two linear operations, performing h' then g' — g'h' ≠ h'g'!).   For functions from vectors to vectors, the chain rule is simply the *product of Jacobians*.   Moreover, as soon as you compose 3 or more functions, it can a make a huge difference whether you multiply the Jacobians from left-to-right ("reverse-mode", or "backpropagation", or "adjoint differentiation") or right-to-left ("forward-mode"). Showed, for example, that if you have *many inputs but a single output* (as is common in machine learning and other types of optimization problem), that it is vastly more efficient to multiply left-to-right than right-to-left, and such "backpropagation algorithms" are a key factor in the practicality of large-scale optimization.  Finally, began talking about functions in more general vector spaces, such as functions with **matrix inputs and/or outputs**.  For example, considered f(A)=A³, giving d(A³)=f′(A)[dA]=A²(dA)+A(dA)A+(dA)A² (≠3A²dA!), and f(A)=A⁻¹, giving d(A⁻¹)=-A⁻¹(dA)A⁻¹.

Part 2: Began going into more detail on matrix-valued functions, and their relationship to the "Jacobian matrix" picture.  Converting f′(A) to a conventional "Jacobian matrix" in such cases requires converting matrices A into column vectors vec(A), a process called "vectorization" of the matrix (by a common convention: simply stacking the matrix by columns).  Linear operators like f′(A)[dA]=AdA+dAA can then be expressed as "ordinary" matrices via [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product).

**Further reading**: The terms "forward-mode" and "reverse-mode" differentiation are most prevalent in [automatic differentiation (AD)](https://en.wikipedia.org/wiki/Automatic_differentiation), which will will cover later in this course. You can find many, many articles online about [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) in neural networks.   There are many other versions of this, e.g. in differential geometry the derivative linear operator (multiplying Jacobians and perturbations dx right-to-left) is called a [pushforward](https://en.wikipedia.org/wiki/Pushforward_(differential)), whereas multiplying a gradient row vector (covector) by a Jacobian left-to-right is called a [pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry)).   This [video on the principles of AD in Julia](https://www.youtube.com/watch?v=UqymrMG-Qi4) by [Dr. Mohamed Tarek](https://github.com/mohamed82008) also starts with a similar left-to-right (reverse) vs right-to-left (forward) viewpoint and goes into how it translates to Julia code, and how you define custom chain-rule steps for Julia AD.


## Lecture 3 (Jan 23)

* part 2: matrix Jacobians via [vectorization](https://en.wikipedia.org/wiki/Vectorization_(mathematics)); notes: [2×2 Matrix Jacobians (html)](https://rawcdn.githack.com/mitmath/matrixcalc/3f6758996e40c5c1070279f89f7f65e76e08003d/notes/2x2Jacobians.jl.html) [(pluto notebook source code)](https://github.com/mitmath/matrixcalc/blob/main/notes/2x2Jacobians.jl)
* part 2: finite-differences ([notes](notes/Finite%20difference%20checks.ipynb))
* video recording (MIT only): [zoom recording](https://mit.zoom.us/rec/share/76jdWTLvGd4ND7cvTpebMTLBwU97iFwap8dG72sAAaEyf6Gnelwi0k1FhVhQo7Qy.AckiCGpbmvVWUP3S?startTime=1674490057000) and [classroom recording](https://mit.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx?folderID=a72e3378-1c0f-425d-ae52-af880157e85c)

Continued from lecture 2: Matrix functions, Jacobians, vectorizations, and Kronecker products.  More examples of matrix functions, including LU factorization and 2×2 eigenproblems.

Finite-difference methods: viewing f(x+δx)–f(x) as an approximation for f'(x)δx on a computer.  This is extremely useful as a quick check of a hand-derived derivative (which is very error prone for complicated functions), and can also be used as a replacement for analytical derivatives in a pinch.  Analyzed two sources of error: truncation error (from the non-infinitesimal δx) and roundoff error (from the finite precision of computer arithmetic).

**Further reading**: Wikipedia has a useful list of [properties of the matrix trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Properties).  The "matrix dot product" introduced today is also called the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product), and the corresponding norm ("length" of the matrix viewed as a vector) is the [Frobenius norm](https://mathworld.wolfram.com/FrobeniusNorm.html).   When you "flatten" a matrix A by stacking its columns into a single vector, the result is called [vec(A)](https://en.wikipedia.org/wiki/Vectorization_(mathematics)), and many important linear operations on matrices can be expressed as [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product).  The [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) has lots of formulas for derivatives of matrix functions.  There is a lot of information online on [finite difference approximations](https://en.wikipedia.org/wiki/Finite_difference),  [these 18.303 notes](https://github.com/mitmath/18303/blob/fall16/difference-approx.pdf), or [Section 5.7 of *Numerical Recipes*](http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf).   The Julia [FiniteDifferences.jl package](https://github.com/JuliaDiff/FiniteDifferences.jl) provides lots of algorithms to compute finite-difference approximations; a particularly robust and powerful way to obtain high accuracy is to employ [Richardson extrapolation](https://github.com/JuliaDiff/FiniteDifferences.jl#richardson-extrapolation) to smaller and smaller δx.  If you make δx too small, the finite precision (#digits) of [floating-point arithmetic](https://en.wikipedia.org/wiki/Floating-point_arithmetic) leads to [catastrophic cancellation errors](https://en.wikipedia.org/wiki/Catastrophic_cancellation).


## Lecture 4 (Jan 23)

* part 1: generalized gradients and inner products — [handwritten notes](https://www.dropbox.com/s/dtdriu0jg1sqoqr/Derivatives%20as%20Linear%20Operators.pdf?dl=0)
* part 2: nonlinear root-finding, optimization, and adjoint-method differentiation [slides](https://docs.google.com/presentation/d/1U1lB5bhscjbxEuH5FcFwMl5xbHl0qIEkMf5rm0MO8uE/edit?usp=sharing)
* video recording (MIT only): [zoom recording](https://mit.zoom.us/rec/share/WmvhY3Z-1LDQZ5MDh68KJO6Rlpya4lUvbhpd0cCIrz3CO6Wk3zz8kLQi_spx8kWf.WhCTDIEHR66GxzSh?startTime=1674662811000) and [classroom recording](https://mit.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx?folderID=a72e3378-1c0f-425d-ae52-af880157e85c)
* [pset 1 solutions](psets/pset1sol.pdf)

**part 0:** To begin with, spent a few minutes talking about the last few sections of the [finite-difference notes](notes/Finite%20difference%20checks.ipynb)) from last lecture: higher-order finite-difference rules, and finite differences in higher dimensions (e.g. for gradients).

**part 1:** Generalizing **gradients** to *scalar* functions f(x) for x in arbitrary *vector spaces* x ∈ V.   The key thing is that we need not just a vector space, but an **inner product** x⋅y (a "dot product", also denoted ⟨x,y⟩ or ⟨x|y⟩); V is then formally called a [Hilbert space](https://en.wikipedia.org/wiki/Hilbert_space).   Then, for *any* scalar function, since df=f'(x)[dx] is a linear operator mapping dx∈V to scalars df∈ℝ (a "[linear form](https://en.wikipedia.org/wiki/Linear_form)"), it turns out that it [*must* be a dot product](https://en.wikipedia.org/wiki/Riesz_representation_theorem) of dx with "something", and we call that "something" the gradient!  That is, once we define a dot product, then for any scalar function f(x) we can define ∇f by f'(x)[dx]=∇f⋅dx.  So ∇f is always something with the same "shape" as x (the [steepest-ascent](https://math.stackexchange.com/questions/223252/why-is-gradient-the-direction-of-steepest-ascent) direction).

Defined the most obvious inner product of m×n matrices: the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product) A⋅B=`sum(A .* B)`=trace(AᵀB)=vec(A)ᵀvec(B), the sum of the products of the matrix entries.  This also gives us the "Frobenius norm" ‖A‖²=A⋅A=trace(AᵀA)=‖vec(A)‖², the square root of the sum of the squares of the entries.   Using this, we can now take the derivatives of various scalar functions of matrices, e.g. we considered

* f(A)=‖A‖ ⥰ ∇f = A/‖A‖
* f(A)=xᵀAy ⥰ ∇f = xyᵀ (for constant x, y)
* f(A)=det(A) ⥰ ∇f = det(A)(A⁻¹)ᵀ = [adjugate](https://en.wikipedia.org/wiki/Adjugate_matrix)(A)ᵀ: we will prove this later

**part 2**: Applications of derivatives to multivariate root-finding and optimization.   A key fact enabling large-scale optimization, i.e. min f(x) where f is a scalar function of *many* parameters x, is that computing ∇f has about the same cost as f, using what is variously called "reverse-mode" or "adjoint" or "backpropagation" differentiation algorithms, which essentially boil down to evaluating the chain rule **left to right**.  Went through a few examples of this, oriented more at engineering/physics optimization (and "topology optimization").

**Further reading (part 2)**:  SGJ gave another [overview of optimization](https://github.com/mitmath/18335/blob/spring21/notes/optimization.pdf) in 18.335 ([video](https://mit.zoom.us/rec/share/QwT0OMMFfkgi9dD0Zoa_3UK-14LbQR8GFcd7Q-O9PqIJTbbULGYqX3isDkLa1kOw.CNhZ0KukrqW2kxeT?startTime=1619031558000)).  There are many textbooks on [nonlinear optimization](http://www.athenasc.com/nonlinbook.html) algorithms of various sorts, including specialized books on [convex optimization](http://web.stanford.edu/~boyd/cvxbook/), [derivative-free optimization](http://bookstore.siam.org/mp08/), etcetera.  A useful review of topology-optimization methods can be found in [Sigmund and Maute (2013)](https://link.springer.com/article/10.1007/s00158-013-0978-6).  See the [notes on adjoint methods](https://github.com/mitmath/18335/blob/spring21/notes/adjoint/adjoint.pdf) and [slides](https://github.com/mitmath/18335/blob/spring21/notes/adjoint/adjoint-intro.pdf) from 18.335 ([video](https://mit.zoom.us/rec/share/xLxMyhBSoIhSFxce5lHb1ubItby5BKFs6mgJJ7kMmjotETmaYm4YA22TA8w8n13i.6sTEFrkkloG7LFeR?startTime=1619463273000)).


## Lecture 4 (Jan 23)

* part 0: norms and derivatives: why a norm of the input and output are needed to *define* a derivative — [handwritten notes](https://www.dropbox.com/s/dtdriu0jg1sqoqr/Derivatives%20as%20Linear%20Operators.pdf?dl=0)
* part 1: [derivative of matrix determinant and inverse](https://rawcdn.githack.com/mitmath/matrixcalc/b08435612045b17745707f03900e4e4187a6f489/notes/determinant_and_inverse.html) [(julia source)](notes/determinant_and_inverse.jl)
* part 2: forward-mode automatic differentiation via dual numbers: notes to be posted
* part 3: forward and reverse-mode automatic differentiation on computational graphs
* video recording (MIT only): zoom recording (coming soon) and [classroom recording](https://mit.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx?folderID=a72e3378-1c0f-425d-ae52-af880157e85c)
* pset 2, coming soon: due Feb 1

**Further reading (part 1)**: There are lots of discussions of the
[derivative of a determinant](https://en.wikipedia.org/wiki/Jacobi%27s_formula) online, involving the ["adjugate" matrix](https://en.wikipedia.org/wiki/Adjugate_matrix) det(A)A⁻¹.
Not as well documented is that the gradient of the determinant is the cofactor matrix widely used for the [Laplace expansion](https://en.wikipedia.org/wiki/Laplace_expansion) of a determinant.
The formula for the [derivative of log(det A)](https://statisticaloddsandends.wordpress.com/2018/05/24/derivative-of-log-det-x/) is also nice, and logs of determinants appear in surprisingly many applications (from statistics to quantum field theory).  The [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) contains many of these formulas, but no derivations.   A nice application of d(det(A)) is solving for eigenvalues λ by applying Newton's method to det(A-λI)=0, and more generally one can solve det(M(λ))=0 for any function Μ(λ) — the resulting roots λ are called [nonlinear eigenvalues](https://en.wikipedia.org/wiki/Nonlinear_eigenproblem) (if M is nonlinear in λ), and one can [apply Newton's method](https://www.maths.manchester.ac.uk/~ftisseur/talks/FT_talk2.pdf) using the determinant-derivative formula here.

**Further reading (part 2)**: Googling "automatic differentiation" will turn up many, many resources — this is a huge practical field these days.   [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) (described detail by [this paper](https://arxiv.org/abs/1607.07892)) in Julia uses [dual number](https://en.wikipedia.org/wiki/Dual_number) arithmetic similar to lecture to compute derivatives; see also this [AMS review article](http://www.ams.org/publicoutreach/feature-column/fc-2017-12), or google "dual number automatic differentiation" for many other reviews.

**Further reading (part 3)**: See [Prof. Edelman's poster](backprop_poster.pdf) about backpropagation on graphs, this blog post on [calculus on computational graphs](https://colah.github.io/posts/2015-08-Backprop/) for a gentle review, and these Columbia [course notes](http://www.cs.columbia.edu/~mcollins/ff2.pdf) for a more formal approach.  Implementing automatic reverse-mode AD is much more complicated than defining a new number type, unfortunately, and involves a lot more intricacies of compiler technology.  See also Chris Rackauckas's blog post on [tradeoffs in AD](https://www.stochasticlifestyle.com/engineering-trade-offs-in-automatic-differentiation-from-tensorflow-and-pytorch-to-jax-and-julia/), and Chris's discussion post on [AD limitations](https://discourse.julialang.org/t/open-discussion-on-the-state-of-differentiable-physics-in-julia/72900/2).