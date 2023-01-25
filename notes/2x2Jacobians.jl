### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 2b24bf38-7398-11ec-2375-3360ec3a2b76
using Symbolics, LinearAlgebra, PlutoUI

# ╔═╡ af4081dd-b957-4c5d-b05f-76b2dc45f982
using ForwardDiff

# ╔═╡ bdbdb5e1-aa07-4724-9bb8-b2472dcf754d
md"""
# Two by Two Matrix Jacobians
"""

# ╔═╡ 01e3eafb-969a-4d86-ae72-9b8451d910ee
md"""
This notebook emphasizes the multiple views of Jacobians with  examples of 2x2 matrix functions.

In particular we will see the
* Symbolic "vec" format producing 4x4 matrices (generally n² by n² or mn by mn)
* Numerical formats
* The important Linear Transformation view
* Kronecker notation
* An example using ForwardDiff automatic differentiation

We also emphasize that  matrix factorizations are also matrix functions, just as much as the square and the cube.
"""

# ╔═╡ 473e7843-2c5c-4602-b0d2-022edc5cb317
TableOfContents(title="Two by Two Matrix Jacobians", indent=true,aside=true)

# ╔═╡ 15626238-97a6-4a9b-8e62-694c75255e18
md"""
# Symbolic Matrices
"""

# ╔═╡ f3df5702-881d-41f2-9788-25cdab6863fb
@variables p,q,r,s,θ

# ╔═╡ df4b65fb-c061-4ff1-bef8-7df3b6dc8cbc
X = [p r;q s]

# ╔═╡ d3682699-c9f3-43d0-a0e6-42500a36a0fe
md"""
## vec
The `vec` command in Julia and in standard mathematics flattens a matrix column by column.
"""

# ╔═╡ a052db18-1aef-4bd2-8f4a-64ee9fe0d93a
vec(X)

# ╔═╡ 22b9fb13-bbb8-4b41-adde-a70969f4b176
md"""
# 1) The matrix square function
"""

# ╔═╡ c8eb1613-1be3-4334-a025-e102d8a0b45a
X^2

# ╔═╡ ade5eaaf-0b53-4230-9264-6c0c7faa3486
vec(X^2)

# ╔═╡ 2e886e5d-b23d-4d72-a14f-c069c6ce416b
md"""
 ## Symbolic Jacobian
The Jacobian of the (flattened) matrix function X² symbolically
"""

# ╔═╡ 2a93a2f0-a846-4c7c-91ab-7da411902022
jac(Y,X) =  Symbolics.jacobian(vec(Y),vec(X))

# ╔═╡ f4789b01-101d-47f3-9dd2-b48fa845d271
J = jac(X^2, X)

# ╔═╡ b647c2d0-07b6-48ef-a3cb-c09e95da2e1f
md"""
## Numerical Jacobian
"""

# ╔═╡ 2507be58-29b5-4b65-b093-c1ba5b1161f3
begin
  M = [1 2;3 4]
  E = [.0003 .0003;.0002 .0001]
  substitute(J,Dict(p=>1,q=>3,r=>2,s=>4))
end

# ╔═╡ f5d34fa1-f673-440d-a6f2-1c978ba77230
substitute(J,Dict(p=>1,q=>3,r=>2,s=>4)) * vec(E)

# ╔═╡ 8d66e0f8-c739-44cf-a280-f03945098eb9
(M+E)^2 - M^2

# ╔═╡ fdb5bb9b-1491-48f6-aa5f-c74835b2e947
md"""
## Linear Transformation Jacobian 
Notice: there is no flattening; this is just matrix to matrix.
"""

# ╔═╡ 82dc6ca6-b232-4721-8132-52628acf2996
linear_transformation(E) = M*E + E*M

# ╔═╡ b217a000-8fbd-4497-a7cb-5172378442bc
linear_transformation(E)

# ╔═╡ 2a373129-5ea1-4345-b55f-2d11c441b929
md"""
## Kronecker product or ⊗ notation
Notation that kind of lets you think "flattened" or "not flattened" at the same time
"""

# ╔═╡ f7942604-85e6-4220-8473-1f0a78e1648e
@variables a,b,c,d

# ╔═╡ d8b6487f-9349-4bc8-a121-6b908ce03b3b
[p r;q s],[a c;b d]

# ╔═╡ 400adebe-26cd-4174-a797-f64d6bc24aa3
md"""
Notice all possible products with the first matrix and the second
"""

# ╔═╡ f1590695-e3ef-4473-b61b-2081401ff05c
kron([a;b],[p q;r s])

# ╔═╡ ee91570e-9668-4da5-9b3d-9dc4407dad32
kron([a c;b d],[p q;r s] )

# ╔═╡ 8ab37951-7f07-4d0e-bd02-b89069807062
@variables e f g h  🍕 👽 🐼 😸

# ╔═╡ 8e16e22b-8a01-4c67-8314-fec376b4d59d
kron([a b c;d e f],[🍕 👽; 🐼 😸])

# ╔═╡ d0192d30-28ca-451e-9d6b-2026fbef757a
md"""
It is very reasonable to express the Jacobian of the matrix square function as $(br)
``I_2 \otimes X + X^T \otimes I_2``
"""

# ╔═╡ df8abae6-14ef-45f9-9545-a5e6ef27e56c
begin
	I2 = [1 0; 0 1]
	kron(I2,X) + kron(X',I2) , J
end

# ╔═╡ 4567d87e-6234-4c46-8b9b-ad3cf7c86150
kron([🍕 👽; 🐼 😸],I2)

# ╔═╡ 1a455f8e-1b43-4037-a8fe-829c23fdb05b
kron(I2,[🍕 👽; 🐼 😸])

# ╔═╡ e0beb540-8e6b-4ed1-86ff-c9460b032b13
kron(I2,X)

# ╔═╡ 056e4b31-b070-492b-9165-3f2a79693184
kron(X',I2)

# ╔═╡ 35139dda-d446-4213-a13e-66a21332beae
md"""
### Key Kronecker identity
  (A ⊗ B) * vec(C) =  vec(BCAᵀ)
"""

# ╔═╡ 0a78fe1c-e613-48b9-8e3b-2fe162c4b238
begin
	A = rand(5,7)
	B = rand(4,3)
	C = rand(3,7)
	kron(A,B) * vec(C) ≈ vec(B*C*A')
end

# ╔═╡ ccccb3c3-a4c8-4600-af5b-62eaff769079
kron( rand(5,5) , rand(5,5) )

# ╔═╡ 12b2e304-a25c-4dfb-882b-5d77b029438f
md"""
 Useful Krockecker identities


* $(A\otimes B)^T=A^T\otimes B^T$
* $(A\otimes B)^{-1}=A^{-1}\otimes B^{-1}$
* $\det(A\otimes B)=\det(A)^m\det(B)^n$, $A\in\Re^{n,n}, B\in\Re^{m,m}$ 
* $trace(A\otimes B)=trace(A)trace(B)$
* $A\otimes B$ is orthogonal if $A$ and $B$ are orthogonal
* $(A \otimes B)(C \otimes D)=(AC) \otimes (BD)$
* If $Au = \lambda u$, and $Bv=\mu v$, then if $X=vu^T$, then
  $BXA^T =\lambda \mu X$, and also $AX^T B^T =
  \lambda \mu X^T$.  Therefore $A \otimes B$ and $B \otimes A$
  have the same eigenvalues, and transposed eigenvectors.

(See [Wikipedia](https://en.wikipedia.org/wiki/Kronecker_product#Properties) for more properties. )
"""

# ╔═╡ 3173f5ad-88bb-48dc-b247-e3f33fbd6f56
md"""
## The Jacobian in Kronecker notation
"""

# ╔═╡ 45864dac-896a-483c-8b8a-c040fb99cebb
md"""
You see (I⊗X + X'⊗I) vec(dX) = vec(XdX + dX X) = vec( d(X²))  $(br)
showing that d(X²) = (I⊗X + X'⊗I) dX.

(I feel it's okay to drop the "vec" and think of the kronecker notation
as defining the linear operator from matrices to matrices)

Do look this over. $(br)
"""

# ╔═╡ 678f07c2-bb5d-4464-ba7b-044dc97719dc
md"""
## Automatic Differentiation (is not finite differences nor symbolic)
It comes in forward and reverse modes. Let's try forward.
"""

# ╔═╡ 4b10b0a3-f47b-493a-a47f-2e30c24847a2
J

# ╔═╡ e9186f6d-485e-446e-b412-1f3de9a9473f
ForwardDiff.jacobian(X->X^2,M)

# ╔═╡ 70fa3888-4fd8-4dae-993f-e9defd565d19
#Check
substitute(J, Dict(X.=>[1 3;2 4] ))

# ╔═╡ 21030b50-6484-4d9b-a58f-5655f769e367
ForwardDiff.jacobian(X->X^2,X)

# ╔═╡ 0fe3ac0b-fa39-4210-b442-2255314b2ea2
md"""
# 2) The matrix cube Function
"""

# ╔═╡ 5f86f7b0-5165-496a-8951-8306825efb5c
expand.(X^3)

# ╔═╡ 5fe9ab2a-9988-4773-bb8d-c2182eafd0d6
md"""
 ## Symbolic Jacobian
The Jacobian of the (flattened) matrix function X² symbolically
"""

# ╔═╡ 9ca4d018-bfd6-4b91-8e98-a4485dae90c0
expand.(jac(X^3, X))

# ╔═╡ c1ec4ac7-7185-4d8e-b6b3-2652912cb3a7
expand.(ForwardDiff.jacobian(X->X^3,X))

# ╔═╡ 6d3f1227-1b9f-4dfd-9c3e-99531a45d8ad
md"""
##  LinearTransformation Jacobian
"""

# ╔═╡ 0f5bc339-d325-4ab4-be75-93f3d7aba92a
md"""
 dX X² + X dX X + dX X²
"""

# ╔═╡ 00514808-b660-4ccb-b3f6-4bfadbde6db0
md"""
with numerical data:
"""

# ╔═╡ 30d07b76-0361-4d47-b8ab-a8c281163683
E -> E*M*M + M*E*M + M*M*E

# ╔═╡ 0612d07e-3752-4e3e-b5ad-a88ba464a25f
(E+M)^3 - M^3

# ╔═╡ 8dcf84c3-20ce-4e7a-976c-c17fe273e118
(E -> E*M*M + M*E*M + M*M*E)(E)

# ╔═╡ d2f3dcd4-cdb0-451c-b5cf-13e68ed79c19
md"""
check against the symbolic answer
"""

# ╔═╡ fd77efc7-fde0-4c40-bba8-6bb2e0e1e3c2
substitute( Symbolics.jacobian(vec(X^3), vec(X)) , Dict(p=>M[1,1],q=>M[2,1],r=>M[1,2],s=>M[2,2]))

# ╔═╡ bc61a209-88f1-437f-a16c-ca3deab4cd38
substitute( Symbolics.jacobian(vec(X^3), vec(X)) , Dict(p=>M[1,1],q=>M[2,1],r=>M[1,2],s=>M[2,2])) * vec(E)

# ╔═╡ a7ea01b6-3347-4bd7-ad27-e067ee103426
md"""
## The Jacobian in Kronecker Notation
"""

# ╔═╡ 5aa421ca-a6a6-447f-8af1-d724cea626ae
expand.( kron(I2,X^2) + kron(X',X) + kron(X'^2,I2) )

# ╔═╡ 3acd6aaf-e40c-4748-8180-57a53368eac6
md"""
# 3) The LU Decomposition

Recall the LU Decomposition factors a matrix into unit lower-trianguar and upper triangular:
"""

# ╔═╡ c28d1043-259f-4045-8148-031da91dc818
begin
	L,U = lu(X);
	L,U
end

# ╔═╡ c621d2fc-9808-4035-9674-21d984043b8e
simplify_fractions.(L*U)

# ╔═╡ e322f5a8-93b4-4d6e-ab93-c98d56cf853e
md"""
The four entries of X: p,q,r,s are transformed into these four entries in LU:
"""

# ╔═╡ 4695b82d-2f06-42e5-b446-da8a7a23046b
[L[2,1],U[1,1],U[1,2],U[2,2]]

# ╔═╡ bf000d5a-ccdb-4fbe-a13f-6277758e9c99
jac([L[2,1],U[1,1],U[1,2],U[2,2]], X)

# ╔═╡ aed69d60-dfc3-421c-900f-20fe4450a64e
md"""
Exercise: Relate this to d(LU) = dL U + L dU
"""

# ╔═╡ 261c7aef-0fad-4e38-ab5d-3a28eda8061a
md"""
# 4) Traceless symmetric eigenproblem: an example with two parameters not four
"""

# ╔═╡ 27b84eb1-54ce-4ad2-9481-0303360ad252
S = [p s; s -p]

# ╔═╡ df2247ff-8748-403f-bb31-87c97a937f3b
md"""
We know that the eigenvalues add to 0 (from the trace) and the eigenvectors are orthogonal (from being symmetric), so we can represent the eigenvectors and eigenvalues:
"""

# ╔═╡ 52cf1b9f-6536-451a-9bec-e7a049773d97
Q = [cos(θ/2) -sin(θ/2); sin(θ/2) cos(θ/2)]  # Eigenvector matrix

# ╔═╡ bce0919f-5951-4f52-96ee-a71fa49c5228
Λ = [r 0;0 -r] # Eigenvalue matrix

# ╔═╡ f14811a4-8e89-4613-b543-e5b85ebd6389
Q

# ╔═╡ e804c01e-d83b-41a4-94ae-8c515b985b1f
Λ

# ╔═╡ 407e8403-6d9f-442c-985a-04e578c3ed0d
Symbolics.simplify.(Q * Λ * Q')

# ╔═╡ c5022a6a-8991-42c0-95b2-fbe0ef1f5773
md"""
The relationship between θ,r to p,s:
"""

# ╔═╡ b25dc0d2-c533-435d-ad43-748b94a8da22
S, simplify.(Q*Λ*Q'), [r*cos(θ) r*sin(θ) ; r*sin(θ) -r*cos(θ)]

# ╔═╡ 58de51b7-6ae3-4fc0-98b5-5e9eba8a0876
simplify.(jac( (Q*Λ*Q')[1:2] ,  [r,θ]))

# ╔═╡ 1281cb63-afd9-419f-bb39-4807def53309
md"""
Interesting mathematical observation: these are the formulas you may remember
from other classes that relate cartesian coordinates to polar coordinates in the plane.
"""

# ╔═╡ 426c7518-150c-4b3a-9fdc-1eb59c48c9cb
jacobian_det = simplify(det(simplify.(jac( (Q*Λ*Q')[1:2] ,  [r,θ]))))

# ╔═╡ e312cf5c-7c05-4cf8-9d63-639cd614e564
md"""
Mathematical aside det J=r , this is the change of variables from x,y to r,θ that you may have seen in 18.02.  This eigenvalue problem is the same as the cartesian coordinates to polar representations of the plane. Often written dx dy = r dr dθ
"""

# ╔═╡ a67c807d-2d65-43fa-944a-7a8e6df031b9
md"""
# 5) The full 2x2 symmetric eigenproblem
"""

# ╔═╡ ad3b7548-8822-4304-a24d-d3e49e79a09f
@variables λ₁ λ₂

# ╔═╡ 1f67e9f6-23ce-43ae-9280-84a5747fa577
md"""
We think of

``\left( \begin{array}{cc}
p & s \\ s & r 
\end{array} \right) =
\left( \begin{array}{rr} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) 
\end{array} \right)
\left( \begin{array}{cc}
\lambda_1 & 0 \\ 0  & \lambda_2 
\end{array} \right) 
\left( \begin{array}{rr} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) 
\end{array} \right)^T 
``
$(br)
as the function from $(br)
``\lambda_1,\lambda_2,θ  \rightarrow  p,r,s``

"""

# ╔═╡ 1b9912e2-3d35-43a6-9d52-6aef7c780873
md"""
 S = QΛQ':
"""

# ╔═╡ 1b640642-920e-4ec3-80e9-9001e96e1634
let 
	Q = [cos(θ) -sin(θ); sin(θ) cos(θ)]
	S = Q*[λ₁ 0;0 λ₂]*Q'
	[p s;s r], S
	J = jac([S[1,1],S[2,2],S[1,2]] , [λ₁,λ₂,θ])
end

# ╔═╡ 03a0fb73-a3f3-4c6f-9483-369beeac85ba
md"""
The determinant of this transformation simplifies to ``\lambda_1 - \lambda_2``
which some people interpret as a kind of repulsion between the two eigenvalues:
that is there is a tendency for the two eigenvalues to not want to be too close
together.  (If both are equal, when n=2, the matrix is ``\alpha I``, one condition
takes three parameters down to 1)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[compat]
ForwardDiff = "~0.10.24"
PlutoUI = "~0.7.29"
Symbolics = "~4.3.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0-rc4"
manifest_format = "2.0"
project_hash = "966c575027181f6f85fc66852aa1872a19e142a0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgCheck]]
git-tree-sha1 = "dedbbb2ddb876f899585c4ec4433265e3017215a"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.1.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "ffc6588e17bcfcaa79dfa5b4f417025e755f83fc"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "4.0.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AutoHashEquals]]
git-tree-sha1 = "45bb6705d93be619b81451bb2006b7ee5d4e4453"
uuid = "15f4f7f2-30c1-5605-9d31-71845cf9641f"
version = "0.2.0"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "a33794b483965bf49deaeec110378640609062b1"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.34"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijections]]
git-tree-sha1 = "705e7822597b432ebe152baa844b49f8026df090"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.3"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "926870acb6cbcf029396f2f2de030282b6bc1941"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.4"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "d5b014b216dc891e81fea299638e4c10c657b582"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.2"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "9bc5dac3c8b6706b58ad5ce24cffd9861f07c94f"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.9.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "97e9e9d0b8303bae296f3bdd1c2b0065dcb7e7ef"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.38"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "StaticArrays", "Statistics"]
git-tree-sha1 = "5f5f0b750ac576bcf2ab1d7782959894b304923e"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.5.9"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DynamicPolynomials]]
deps = ["DataStructures", "Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "585de0d658506cf0fe5808026edff662bef5bf03"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.4.1"

[[deps.EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "d7ab55febfd0907b285fbf8dc0c73c0825d9d6aa"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.3.0"

[[deps.ExprTools]]
git-tree-sha1 = "24565044e60bc48a7562e75bcf14f084901dc0b6"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.7"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2b72a5624e289ee18256111657663721d59c143e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.24"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "22df5b96feef82434b07327e2d3c770a9b21e023"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "LinearAlgebra", "MacroTools", "StaticArrays"]
git-tree-sha1 = "41158dee1d434944570b02547d404e075da15690"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.7.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Metatheory]]
deps = ["AutoHashEquals", "DataStructures", "Dates", "DocStringExtensions", "Parameters", "Reexport", "TermInterface", "ThreadsX", "TimerOutputs"]
git-tree-sha1 = "0886d229caaa09e9f56bcf1991470bd49758a69f"
uuid = "e9d8d322-4543-424a-9be4-0cc815abe26c"
version = "1.3.3"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariatePolynomials]]
deps = ["DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "fa6ce8c91445e7cd54de662064090b14b1089a6d"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.4.2"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "73deac2cbae0820f43971fad6c08f6c4f2784ff2"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.3.2"

[[deps.NaNMath]]
git-tree-sha1 = "f755f36b19a5116bb580de457cda0c140153f283"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.6"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "92f91ba9e5941fc781fecf5494ac1da87bdac775"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "5c0eb9099596090bb3215260ceca687b888a1575"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.30"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "6b96eb51a22af7e927d9618eaaf135a3520f8e2f"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.24.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "e681d3bfa49cd46c3c161505caddf20f0e62aaa9"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "cdc1e4278e91a6ad530770ebb327f9ed83cf10c4"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.3"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "40c1c606543c0130cd3673f0dd9e11f2b5d76cd0"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.26.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "0afd9e6c623e379f593da01f20590bacc26d1d14"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e08890d19787ec25029113e88c34ec20cac1c91e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.0.0"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "b4912cd034cdf968e06ca5f943bb54b17b97793a"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.5.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "2ae4fe21e97cd13efd857462c1869b73c9f61be3"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "bedb3e17cc1d94ce0e6e66d3afa47157978ba404"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.14"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "Metatheory", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "TermInterface", "TimerOutputs"]
git-tree-sha1 = "3f8f28a4d36f224bb3f79ddc5b675b78cec2e16b"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "0.19.2"

[[deps.Symbolics]]
deps = ["ArrayInterface", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "IfElse", "Latexify", "Libdl", "LinearAlgebra", "MacroTools", "Metatheory", "NaNMath", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicUtils", "TermInterface", "TreeViews"]
git-tree-sha1 = "074e08aea1c745664da5c4b266f50b840e528b1c"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "4.3.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TermInterface]]
git-tree-sha1 = "7aa601f12708243987b88d1b453541a75e3d8c7a"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "0.2.3"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadsX]]
deps = ["ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "Setfield", "SplittablesBase", "Transducers"]
git-tree-sha1 = "6dad289fe5fc1d8e907fa855135f85fb03c8fa7a"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.9"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "a5aed757f65c8a1c64503bc4035f704d24c749bf"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.14"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "3f0945b47207a41946baee6d1385e4ca738c25f7"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.68"

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─bdbdb5e1-aa07-4724-9bb8-b2472dcf754d
# ╟─01e3eafb-969a-4d86-ae72-9b8451d910ee
# ╠═2b24bf38-7398-11ec-2375-3360ec3a2b76
# ╠═473e7843-2c5c-4602-b0d2-022edc5cb317
# ╟─15626238-97a6-4a9b-8e62-694c75255e18
# ╠═f3df5702-881d-41f2-9788-25cdab6863fb
# ╠═df4b65fb-c061-4ff1-bef8-7df3b6dc8cbc
# ╟─d3682699-c9f3-43d0-a0e6-42500a36a0fe
# ╠═a052db18-1aef-4bd2-8f4a-64ee9fe0d93a
# ╟─22b9fb13-bbb8-4b41-adde-a70969f4b176
# ╠═c8eb1613-1be3-4334-a025-e102d8a0b45a
# ╠═ade5eaaf-0b53-4230-9264-6c0c7faa3486
# ╟─2e886e5d-b23d-4d72-a14f-c069c6ce416b
# ╠═2a93a2f0-a846-4c7c-91ab-7da411902022
# ╠═f4789b01-101d-47f3-9dd2-b48fa845d271
# ╟─b647c2d0-07b6-48ef-a3cb-c09e95da2e1f
# ╠═2507be58-29b5-4b65-b093-c1ba5b1161f3
# ╠═f5d34fa1-f673-440d-a6f2-1c978ba77230
# ╠═8d66e0f8-c739-44cf-a280-f03945098eb9
# ╟─fdb5bb9b-1491-48f6-aa5f-c74835b2e947
# ╠═82dc6ca6-b232-4721-8132-52628acf2996
# ╠═b217a000-8fbd-4497-a7cb-5172378442bc
# ╟─2a373129-5ea1-4345-b55f-2d11c441b929
# ╠═f7942604-85e6-4220-8473-1f0a78e1648e
# ╠═d8b6487f-9349-4bc8-a121-6b908ce03b3b
# ╟─400adebe-26cd-4174-a797-f64d6bc24aa3
# ╠═f1590695-e3ef-4473-b61b-2081401ff05c
# ╠═ee91570e-9668-4da5-9b3d-9dc4407dad32
# ╠═8ab37951-7f07-4d0e-bd02-b89069807062
# ╠═8e16e22b-8a01-4c67-8314-fec376b4d59d
# ╠═4567d87e-6234-4c46-8b9b-ad3cf7c86150
# ╠═1a455f8e-1b43-4037-a8fe-829c23fdb05b
# ╠═e0beb540-8e6b-4ed1-86ff-c9460b032b13
# ╠═056e4b31-b070-492b-9165-3f2a79693184
# ╟─d0192d30-28ca-451e-9d6b-2026fbef757a
# ╠═df8abae6-14ef-45f9-9545-a5e6ef27e56c
# ╟─35139dda-d446-4213-a13e-66a21332beae
# ╠═0a78fe1c-e613-48b9-8e3b-2fe162c4b238
# ╠═ccccb3c3-a4c8-4600-af5b-62eaff769079
# ╟─12b2e304-a25c-4dfb-882b-5d77b029438f
# ╟─3173f5ad-88bb-48dc-b247-e3f33fbd6f56
# ╟─45864dac-896a-483c-8b8a-c040fb99cebb
# ╟─678f07c2-bb5d-4464-ba7b-044dc97719dc
# ╠═af4081dd-b957-4c5d-b05f-76b2dc45f982
# ╠═4b10b0a3-f47b-493a-a47f-2e30c24847a2
# ╠═e9186f6d-485e-446e-b412-1f3de9a9473f
# ╠═70fa3888-4fd8-4dae-993f-e9defd565d19
# ╠═21030b50-6484-4d9b-a58f-5655f769e367
# ╟─0fe3ac0b-fa39-4210-b442-2255314b2ea2
# ╠═5f86f7b0-5165-496a-8951-8306825efb5c
# ╟─5fe9ab2a-9988-4773-bb8d-c2182eafd0d6
# ╠═9ca4d018-bfd6-4b91-8e98-a4485dae90c0
# ╠═c1ec4ac7-7185-4d8e-b6b3-2652912cb3a7
# ╟─6d3f1227-1b9f-4dfd-9c3e-99531a45d8ad
# ╟─0f5bc339-d325-4ab4-be75-93f3d7aba92a
# ╟─00514808-b660-4ccb-b3f6-4bfadbde6db0
# ╠═30d07b76-0361-4d47-b8ab-a8c281163683
# ╠═0612d07e-3752-4e3e-b5ad-a88ba464a25f
# ╠═8dcf84c3-20ce-4e7a-976c-c17fe273e118
# ╟─d2f3dcd4-cdb0-451c-b5cf-13e68ed79c19
# ╠═fd77efc7-fde0-4c40-bba8-6bb2e0e1e3c2
# ╠═bc61a209-88f1-437f-a16c-ca3deab4cd38
# ╟─a7ea01b6-3347-4bd7-ad27-e067ee103426
# ╠═5aa421ca-a6a6-447f-8af1-d724cea626ae
# ╟─3acd6aaf-e40c-4748-8180-57a53368eac6
# ╠═c28d1043-259f-4045-8148-031da91dc818
# ╠═c621d2fc-9808-4035-9674-21d984043b8e
# ╟─e322f5a8-93b4-4d6e-ab93-c98d56cf853e
# ╠═4695b82d-2f06-42e5-b446-da8a7a23046b
# ╠═bf000d5a-ccdb-4fbe-a13f-6277758e9c99
# ╠═aed69d60-dfc3-421c-900f-20fe4450a64e
# ╟─261c7aef-0fad-4e38-ab5d-3a28eda8061a
# ╠═27b84eb1-54ce-4ad2-9481-0303360ad252
# ╟─df2247ff-8748-403f-bb31-87c97a937f3b
# ╠═52cf1b9f-6536-451a-9bec-e7a049773d97
# ╠═bce0919f-5951-4f52-96ee-a71fa49c5228
# ╠═f14811a4-8e89-4613-b543-e5b85ebd6389
# ╠═e804c01e-d83b-41a4-94ae-8c515b985b1f
# ╠═407e8403-6d9f-442c-985a-04e578c3ed0d
# ╟─c5022a6a-8991-42c0-95b2-fbe0ef1f5773
# ╠═b25dc0d2-c533-435d-ad43-748b94a8da22
# ╠═58de51b7-6ae3-4fc0-98b5-5e9eba8a0876
# ╟─1281cb63-afd9-419f-bb39-4807def53309
# ╠═426c7518-150c-4b3a-9fdc-1eb59c48c9cb
# ╟─e312cf5c-7c05-4cf8-9d63-639cd614e564
# ╟─a67c807d-2d65-43fa-944a-7a8e6df031b9
# ╠═ad3b7548-8822-4304-a24d-d3e49e79a09f
# ╟─1f67e9f6-23ce-43ae-9280-84a5747fa577
# ╟─1b9912e2-3d35-43a6-9d52-6aef7c780873
# ╠═1b640642-920e-4ec3-80e9-9001e96e1634
# ╟─03a0fb73-a3f3-4c6f-9483-369beeac85ba
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
