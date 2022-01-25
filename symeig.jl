### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ 7f7e5188-7c86-11ec-34ca-3347d6c19595
using LinearAlgebra, PlutoUI

# ╔═╡ c6386d30-fc92-4d12-922c-3fe14d59ad68
TableOfContents(title="Symmetric Eigenvalue Derivatives", indent=true, depth=4, aside=true)

# ╔═╡ fd24a55b-d4d4-4bfe-a485-552c69c2c7f7
md"""
## Warmup:Differentiating on the unit sphere in n dimensions

Geometrically, we all know that velocity vectors (equivalently tangents) on the sphere are orthogonal to radii. Our differentials say this algebraically:

"""

# ╔═╡ 71ca8eac-0864-4beb-9f1a-b12ef4e6cf7b
begin
	x = randn(5)
	dx= .000001 * randn(5)
	
	q = normalize(x)   # make x a unit vector   x/norm(x)
	dq = normalize(x+dx)-q # make (x+dx) a unit vector and subtract from q
	
	q'dq
end

# ╔═╡ 63df3a83-cd11-4d48-a24a-28d97deea00f
md"""
Since ``x^Tx=1``, we have that $(br)
``2x^Tdx=d(1)=0``, which says that at the point ``x`` on the sphere (a radius, if you will), ``dx``,
the linearization of the constraint of moving along the sphere satisfies ``dx \perp x``
(dot product is 0).

This is our first example where we have seen the infinitesimal perturbation ``dx`` being constrained.
"""

# ╔═╡ 38f7d4de-1072-4127-bbce-4e80da9273a8
md"""
## Special case: a circle
Let us consider simply the circle in the plane.

``x = (\cos \theta, \sin \theta)`` $(br)
``x^T dx = (\cos \theta, \sin \theta) \cdot  (-\sin \theta, \cos \theta) d\theta= 0``

We can think of ``x`` as "extrinsic" coordinates, in that it is a vector in ``R^2.``
On the other hand ``\theta`` is an "intrinsic" coordinate, every point on the circle is specified by one ``\theta``.
"""

# ╔═╡ 44471684-8f86-43b3-8d79-660a48912dc0


# ╔═╡ 5c83619a-3983-4c0a-ac2a-3f88fa6366c5
md"""
Suppose ``A`` is symmetric.  We then know that  if we allow general ``dx`` then 
$(br) ``d(\frac{1}{2}x^TAx)= (Ax)^Tdx `` and we would conclude ``Ax`` is the gradient.
$(br) Now we wish to restrict to the sphere. $(br)

### On the sphere

You may remember that ``I-xx^T`` is a *Projection Matrix* ( meaning that its equal to its square and it is symmetric).  Geometrically the matrix removes components in the ``x`` direction. $(br)  In particular if ``x^Tdx=0``, ``(I-xx^T)dx=dx``.

It follows that if ``x^Tdx=0`` then ``x^TA(dx) = x^TA(I-xx^T)dx = ((I-xx^T)Ax)^T dx``
so that $(I-xx^T)Ax$ is the gradient of ``\frac{1}{2}x^TAx`` on the sphere.

### What did we just do?

To get the gradient we needed two things:
* A linearization of the function that is correct on tangents and
* A direction that is tangent (satisifes the linearized constraint)

### Gradient of a general scalar function on the sphere:

``df= g(x)^T dx = ((I-xx^T)g(x))^Tdx``

Project the unconstrainted gradient to the sphere to get the constrained gradient.  It is the direction of maximal increase on the sphere.


"""

# ╔═╡ ac64ea62-e6f5-4b0b-8a21-4a0d84c9ed7b
md"""
# Differentiating nxn orthogonal matrices (the orthogonal group)
"""

# ╔═╡ 6f687dd6-78fb-4e5f-9bf8-e41f4a7665f4
begin
	A = randn(5,5)
	dA = .00001 * randn(5,5)
	Q = qr(A).Q
	dQ = qr(A+dA).Q - Q
	(Q'dQ)/.00001
end

# ╔═╡ c9ccb3f3-70c6-4f99-92f4-a8529705b237
md"""
Do you see the structure?

Q^TdQ is anti-symmetric (sometimes called skew-symmetric).  

(If ``M=-M^T``, we say that ``M`` is anti-symmetric.  Note all anti-symmetric
have 0 diagonally)

Proof:  The constraint of being orthogonal is ``Q^TQ=I``
so differentiating, ``Q^TdQ + dQ^TQ=0`` which is the same as
saying `` (Q^TdQ) + (Q^TdQ)^T = 0``, but this is the equation
for being antisymmetric.
"""

# ╔═╡ dd659a37-3235-4f56-921a-b348e233ce73
md"""
## What is the dimension of the "surface" of orthogonal matrices in the ``n^2`` dimensional , n by n matrix space?

For example when n=2 we have rotations (and reflections). Rotations have the form
$(br)
``Q = \begin{pmatrix} \cos \theta & \sin \theta \\ -\sin \theta & \cos \theta \end{pmatrix} ``

When n=2 we have one parameter.

When n=3, airplane pilots know about "roll, pitch, and yaw" and these are three parameters. 

For general ``n`` the answer is ``n(n-1)/2.``

A few ways to see that:

* n^2  free parameters, orthogonality ``Q^TQ=I`` imposes n(n+1)/2 constraints leaving
``n(n-1)/2`` free parameters.

* When we do QR, the R "eats" up n(n+1)/2 parameters leaving n(n-1)/2 for Q.

* Think about the symmetric eigenvalue problem:  S = QΛQᵀ.
S has n(n+1)/2 and Λ has n, leaving n(n-1)/2 for Q.

* Think about the singular value decomposition.  A = UΣVᵀ
A has n^2, and Σ has n, leaving n(n-1) to be split evenly for the
orthogonal matrices U and V.

"""

# ╔═╡ acaccf1f-cc17-47d8-ad6d-49108f067e17
md"""
## Differentiating the Symmetric Eigendecomposition
"""

# ╔═╡ 1023e972-bd69-45ae-974a-1a093a57ece4
md"""
`` S = Q \Lambda Q^T`` is the eigendecomposition of a symmetric ``S`` with ``\Lambda`` diagonal containing eigenvalues, and ``Q`` othogonal with columns as eigenvectors.
$(br)

``dS = dQ \Lambda Q^T + Q d\Lambda Q^T + Q Λ dQ^T`` which may be written
``Q^T dS Q = Q^T dQ \Lambda - \Lambda Q^T dQ + d\Lambda``

$(br)

Exercise: Check that the left and right side of the above are both symmetric.
"""

# ╔═╡ 7f566eab-a296-4666-8b5b-e94cf2debf68
let
   A = randn(5,5)
   dA = .00001 * randn(5,5)
   S = A+A' # symmetrize A 
   dS = dA + dA' # symmetrize dA
   Λ,Q = eigen(Symmetric(S))
   Λ₁,Q₁ =	eigen(Symmetric(S+dS)) 
	dQ =  Q₁-Q
	dΛ = Λ₁-Λ 

	
	[Q'*dS*Q  ; diagm(dΛ) + Q'dQ*diagm(Λ)-diagm(Λ)*Q'dQ]

	


end

# ╔═╡ 049e54a1-ee97-4fda-a9c6-2865ffe938a2
md"""
 Maybe easier if one looks at the diagonal entries on their own: $(br)

``(Q^T dS Q)_{ii} = q_i^T dS q_i,`` where ``q_i`` is the ith eigenvector. $(br)
Hence ``q_i^T dS q_i = d\lambda_i.``

Sometimes we think of a curve of matrices ``S(t)`` depending on a parameter such as time.  If we ask for ``\frac{d\lambda_i}{dt}`` we have that it equals ``q_i^T \frac{dS(t)}{dt} q_i``.

How do we get the gradient  ``\nabla \lambda_i`` of one eigenvalue ``\lambda_i``?

trace(``(q_i q_i^T)^T dS``) ``= d\lambda_i``, thus we instantly see
that ``\nabla \lambda_i = q_i q_i^T``
"""

# ╔═╡ 769d7ff6-bae1-423e-890b-69e0a0a9a79b
md"""
What about the eigenvectors? Those come from the off-diagonal elements:
$(br)
``(Q^T dS Q)_{ij} = (Q^T \frac{dQ}{dt})_{ij}(\lambda_j-\lambda_i),`` if ``(i\ne j)``,
so we can form the elements of `` Q^T \frac{dQ}{dt}`` (remember the diagonal is 0),
and left multiply by ``Q`` to obtain  ``\frac{dQ}{dt}.``
"""

# ╔═╡ c9c7011a-8eca-4169-b890-97d6bb4a8244
md"""
It is interesting to get the second derivative of eigenvalues when moving along a line in symmetric matrix space.  For simplicity we'll start at a diagonal matrix ``\Lambda``.

Let ``S(t)= \Lambda + t E``.  

Differentiating ``\frac{d\Lambda}{dt} = diag( Q^T \frac{dS}{dt} Q)`` we get
``\frac{d^2\Lambda}{dt^2} = diag( Q^T \frac{d^2S}{dt^2} Q) + 2 diag(Q^T \frac{dS}{dt} \frac{dQ}{dt}).``
"""

# ╔═╡ c3ae2756-5a33-4ec1-8323-262d65f018c6
md"""
Evaluating at ``Q=I`` and recognizing that the first term is $0$ since we are on a line, we have $(br)

``\frac{d^2\Lambda}{dt^2} = 2 diag( E \frac{dQ}{dt} )``

or

``\frac{d^2\lambda_i}{dt^2} = 2 \sum_{k \ne i} E_{ik}^2/(\lambda_i-\lambda_k).``
"""

# ╔═╡ 81be4b3e-b7b8-4814-8239-11f949561868
md"""
We can write this as a Taylor series.

``\lambda_i(\epsilon) = \lambda_i + \epsilon E_{ii} + \epsilon^2 \sum_{k \ne i} E_{ik}^2/(\lambda_i-\lambda_k) + \ldots ``
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.30"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "92f91ba9e5941fc781fecf5494ac1da87bdac775"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "5c0eb9099596090bb3215260ceca687b888a1575"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.30"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═7f7e5188-7c86-11ec-34ca-3347d6c19595
# ╠═c6386d30-fc92-4d12-922c-3fe14d59ad68
# ╟─fd24a55b-d4d4-4bfe-a485-552c69c2c7f7
# ╠═71ca8eac-0864-4beb-9f1a-b12ef4e6cf7b
# ╟─63df3a83-cd11-4d48-a24a-28d97deea00f
# ╟─38f7d4de-1072-4127-bbce-4e80da9273a8
# ╠═44471684-8f86-43b3-8d79-660a48912dc0
# ╟─5c83619a-3983-4c0a-ac2a-3f88fa6366c5
# ╟─ac64ea62-e6f5-4b0b-8a21-4a0d84c9ed7b
# ╠═6f687dd6-78fb-4e5f-9bf8-e41f4a7665f4
# ╟─c9ccb3f3-70c6-4f99-92f4-a8529705b237
# ╟─dd659a37-3235-4f56-921a-b348e233ce73
# ╟─acaccf1f-cc17-47d8-ad6d-49108f067e17
# ╟─1023e972-bd69-45ae-974a-1a093a57ece4
# ╠═7f566eab-a296-4666-8b5b-e94cf2debf68
# ╟─049e54a1-ee97-4fda-a9c6-2865ffe938a2
# ╟─769d7ff6-bae1-423e-890b-69e0a0a9a79b
# ╟─c9c7011a-8eca-4169-b890-97d6bb4a8244
# ╟─c3ae2756-5a33-4ec1-8323-262d65f018c6
# ╟─81be4b3e-b7b8-4814-8239-11f949561868
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
