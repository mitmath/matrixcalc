# Matrix Calculus for Machine Learning and Beyond

This is the course page for an **18.063 Matrix Calculus** at MIT taught in **January 2026** ([IAP](https://elo.mit.edu/iap/)) by
Professors [Alan Edelman](https://math.mit.edu/~edelman/) and [Steven G. Johnson](https://math.mit.edu/~stevenj/).

* For past versions of this course, see [Matrix Calculus in IAP 2023 (OCW)](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/) on OpenCourseWare (also [on github](https://github.com/mitmath/matrixcalc/tree/iap2023), with videos [on YouTube](https://www.youtube.com/playlist?list=PLUl4u3cNGP62EaLLH92E_VCN4izBKK6OE)).  See also [Matrix Calculus in IAP 2022 (OCW)](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2022/pages/lecture-notes-and-readings/) (also [on github](https://github.com/mitmath/matrixcalc/tree/iap2022)), and [Matrix Calculus 2024 (github)](https://github.com/mitmath/matrixcalc/tree/iap2024) and [Matrix Calculus 2025 (github)](https://github.com/mitmath/matrixcalc/tree/iap2025); some previous years used the temporary 18.S096 "special subject" course number.

**Lectures:** MWF time 11am–1pm, Jan 12–Jan 30 (except Jan 19) in room 2-131.  3 units, *2 problem sets* due Jan 23 and Jan 30 — submitted electronically [via Canvas](https://canvas.mit.edu/courses/29776), no exams.  TA/grader: TBD.

**Course Notes**: [18.063 COURSE NOTES](https://www.dropbox.com/scl/fi/iq4plt8oqja845cuuosa4/Matrix-Calculus-latest.pdf?rlkey=nsnytdu28jje41nhh1bl2dbba&st=i6lfha0r&dl=0).  Other materials to be posted below.

**Piazza forum:** Online discussions at [Piazza](https://piazza.com/mit/spring2025/18063).

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

 Reviewed the (easy) derivations of the sum rule d(f+g)=df+dg and the product rule d(fg) = (df)g+f(dg), directly from the definition of f(x+dx)-f(x)=df=f′(x)[dx], dropping higher-order terms.

 Discussed the chain rule for f(g(x)) (f'(x)=g'(h(x))h'(x), where this is a *composition* of two linear operations, performing h' then g' — g'h' ≠ h'g'!).   For functions from vectors to vectors, the chain rule is simply the *product of Jacobians*.   Moreover, as soon as you compose 3 or more functions, it can a make a huge difference whether you multiply the Jacobians from left-to-right ("reverse-mode", or "backpropagation", or "adjoint differentiation") or right-to-left ("forward-mode"). Showed, for example, that if you have *many inputs but a single output* (as is common in machine learning and other types of optimization problem), that it is vastly more efficient to multiply left-to-right than right-to-left, and such "backpropagation algorithms" are a key factor in the practicality of large-scale optimization.

**Further reading**: *Course Notes* (link above), chapters 1 and 2.
 [matrixcalculus.org](http://www.matrixcalculus.org/) (linked in the slides) is a fun site to play with derivatives of matrix and vector functions.  The [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) has a lot of formulas for these derivatives, but no derivations.  Some [notes on vector and matrix differentiation](https://cdn-uploads.piazza.com/paste/j779e63owl53k6/04b2cb8c2f300212d723bea822a6b856085b28e28ca9debc75a05761a436499c/6.S087_Lecture_2.pdf) were posted for 6.S087 from IAP 2021.

 **Further reading (gradients)**: A fancy name for a row vector is a "covector" or [linear form](https://en.wikipedia.org/wiki/Linear_form), and the fancy version of the relationship between row and column vectors is the [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem), but until you get to non-Euclidean geometry you may be happier thinking of a row vector as the transpose of a column vector.

**Further reading (chain rule)**: The terms "forward-mode" and "reverse-mode" differentiation are most prevalent in [automatic differentiation (AD)](https://en.wikipedia.org/wiki/Automatic_differentiation), which will will cover later in this course. You can find many, many articles online about [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) in neural networks.   There are many other versions of this, e.g. in differential geometry the derivative linear operator (multiplying Jacobians and perturbations dx right-to-left) is called a [pushforward](https://en.wikipedia.org/wiki/Pushforward_(differential)), whereas multiplying a gradient row vector (covector) by a Jacobian left-to-right is called a [pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry)).   This [video on the principles of AD in Julia](https://www.youtube.com/watch?v=UqymrMG-Qi4) by [Dr. Mohamed Tarek](https://github.com/mohamed82008) also starts with a similar left-to-right (reverse) vs right-to-left (forward) viewpoint and goes into how it translates to Julia code, and how you define custom chain-rule steps for Julia AD.  In other fields, "reverse mode" is sometimes called an "adjoint method": see the [notes on adjoint methods](https://github.com/mitmath/18335/blob/spring21/notes/adjoint/adjoint.pdf) and [slides](https://github.com/mitmath/18335/blob/spring21/notes/adjoint/adjoint-intro.pdf) from 18.335 ([video](https://mit.zoom.us/rec/share/xLxMyhBSoIhSFxce5lHb1ubItby5BKFs6mgJJ7kMmjotETmaYm4YA22TA8w8n13i.6sTEFrkkloG7LFeR?startTime=1619463273000)).

**Further reading (fancier math)**: the perspective of derivatives as linear operators is sometimes called a [Fréchet derivative](https://en.wikipedia.org/wiki/Fr%C3%A9chet_derivative) and you can find lots of very abstract (what I'm calling "fancy") presentations of this online, chock full of weird terminology whose purpose is basically to generalize the concept to weird types of vector spaces.  The "little-o notation" o(δx) we're using here for "infinitesimal asymptotics" is closely related to the [asymptotic notation](https://en.wikipedia.org/wiki/Big_O_notation) used in computer science, but in computer science people are typically taking the limit as the argument (often called "n") becomes very *large* instead of very small.
