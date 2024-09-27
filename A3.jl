### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ dc004086-db6e-4813-841e-d427520402f7
begin
	using CSV, DataFrames, StatsPlots, PlutoUI, Random, Statistics
	using LinearAlgebra: dot, norm, norm1, norm2, I
	using Distributions: Distributions, Uniform, TDist, cdf, Normal
	using MultivariateStats: MultivariateStats, PCA
	using StatsBase: StatsBase
	
end

# ╔═╡ 75441ce6-2137-4fcf-bba2-6ed67b9acb59
begin
	_check_complete(complete) = complete ? "✅" : "❌"
	
	md"""
	# Setup

	this section loads and installs all the packages. You should be setup already from assignment 1, but if not please read and follow the `instructions.md` for further details.
	"""
end

# ╔═╡ c1b9d7a3-5605-4478-b1a9-a9553ad38ed7
PlutoUI.TableOfContents(title="A4 Outline")

# ╔═╡ c9797979-7d16-4ba4-8f82-c7ec33f6e517
plotly() # In this notebook we use the plotly backend for Plots.

# ╔═╡ 693a3933-c1c2-4249-8c03-f5151267222f
md"""
### !!!IMPORTANT!!!

Insert your details below. You should see a green checkmark.
"""

# ╔═╡ def97306-1703-42bc-bc09-da623c545e87
student = (name="Khevish Singh Jankee", email="jankee@ualberta.ca", ccid="jankee", idnumber=1692436)

# ╔═╡ bc0c5abe-1c2a-4e5c-bb0c-8d76b5e81133
let
	def_student = (name="NAME as in eclass", email="UofA Email", ccid="CCID", idnumber=0)
	if length(keys(def_student) ∩ keys(student)) != length(keys(def_student))
		md"You don't have all the right entries! Make sure you have `name`, `email`, `ccid`, `idnumber`. ❌"
	elseif any(getfield(def_student, k) == getfield(student, k) for k in keys(def_student))
		md"You haven't filled in all your details! ❌"
	elseif !all(typeof(getfield(def_student, k)) === typeof(getfield(student, k)) for k in keys(def_student))
		md"Your types seem to be off: `name::String`, `email::String`, `ccid::String`, `idnumber::Int`"
	else
		md"Welcome $(student.name)! ✅"
	end
end

# ╔═╡ 14c30b50-157f-40bf-b0a6-0232aa2705c6
md"""
Important Note: You should only write code in the cells that has: """


# ╔═╡ 4a196e82-fe94-49fe-ab87-d57e2d7fbd34
#### BEGIN SOLUTION


#### END SOLUTION

# ╔═╡ 7247e88f-e9c3-4561-ae67-71eb23ad066d
function check_elements(Φ_true::Matrix{Float64}, Φ::Matrix{Float64})
	all_exist = true
	for n in eachindex(Φ_true), m in eachindex(Φ_true[n])
		if Φ_true[n][m] ∉ Φ || Φ[n][m] ∉ Φ_true
			has_sim = false
			for i in eachindex(Φ_true), j in eachindex(Φ_true[i])
				if Φ_true[i][j] ≈ Φ[n][m]
					has_sim = true
				end 
			end 
			true_has_sim = false
			for i in eachindex(Φ), j in eachindex(Φ[i])
				if Φ[i][j] ≈ Φ_true[n][m]
					true_has_sim = true
				end 
			end 
			if has_sim == false || true_has_sim == false
				all_exist = false
			end
		end
	end
	all_exist
end 


# ╔═╡ a7aecd21-13f2-4cf2-8239-3a3d708602c9
md"""
# Q1: Binary Classification

So far, we have only considered algorithms for regression where the target is continuous. We will now explore implementations of algorithms for binary classification.

As before, we have broken our ML systems into smaller pieces. This will allow us to more easily take advantage of code we've already written, and will be more useful as we expand the number of algorithms we consider. We make several assumptions to simplify the code, but the general type hierarchy can be used much more broadly.

We split each system into:
- Model
- Gradient descent procedure
- Loss Function
- Optimization Strategy
"""

# ╔═╡ e3c3e903-e2c2-4de0-a3b2-56a27709e8c3
md"""
## Baselines
The only baseline we will use in this assignment is a random classifier. 
"""

# ╔═╡ a35944ae-7bbb-4f3c-bbe0-d2bbb57cd70b
md"""
### RandomModel
"""

# ╔═╡ 7b513344-1cad-4eef-9faf-e77ba176323e
md"""
# Models

"""

# ╔═╡ 4f4029a2-c590-4bd3-a0db-d2380d4b4620
md"""
## The model interface

- `AbstractModel`: This is an abstract type which is used to derive all the model types in this assignment
- `predict`: This takes a matrix of samples and returns the prediction doing the proper data transforms.
- `get_features`: This transforms the features according to the non-linear transform of the model (which is the identity for linear).
- `get_linear_model`: All models are based on a linear model with transformed features, and thus have a linear model.
- `copy`: This returns a new copy of the model.
"""

# ╔═╡ dcfecc35-f25c-4856-8429-5c31d94d0a42
"""
	AbstractModel

Used as the root for all models in this notebook. We provide a helper `predict` function for `AbstractVectors` which transposes the features to a row vector. We also provide a default `update_transform!` which does nothing.
"""
abstract type AbstractModel end

# ╔═╡ d45dedf8-e46a-4ffb-ab45-f9d211d3a8ca
predict(alm::AbstractModel, x::AbstractVector) = predict(alm, x')[1];

# ╔═╡ 7cd46d84-a74c-44dc-8339-68010924bc39
update_transform!(AbstractModel, args...) = nothing;

# ╔═╡ fd4df2c6-fbfe-4089-a487-e3965a463ef3
md"
#### Linear Model

As before, we define a linear model that inputs a vector $\mathbf{x}$ and outputs prediction $\mathbf{x}^\top\mathbf{w}$. We can also input the whole data matrix $$\mathbf{X}$$ and output a prediction for each row using $$\mathbf{X} \mathbf{w}$$. 
We exploit this in `predict`, to return predictions for the data matrix $$\mathbf{X}$$ of size `(samples, features)`.

We define `get_features`, which we will need for polynomial regression. For logistic regression, the default is to return the inputs themselves. In polynomial logistic regression, we will replace this function with one that returns polynomial features.

"

# ╔═╡ 2d43a3ba-2a2c-4114-882f-5834d42e302a
begin
	struct LinearModel <: AbstractModel
		w::Matrix{Float64} # Aliased to Array{Float64, 2}
	end
	
	LinearModel(in, out=1) = 
		LinearModel(zeros(in, out)) # feature size × output size
	
	Base.copy(lm::LinearModel) = LinearModel(copy(lm.w))
	predict(lm::LinearModel, X::AbstractMatrix) = X * lm.w
	get_features(m::LinearModel, x) = x

end;

# ╔═╡ 9c0491db-ad27-46fc-969c-c42a14cdadeb
md"
#### Logistic Regression Model
Logistic regression is very similar to linear regression. But, unlike linear regression where $y$ is a continuous variable, the targets are binary $y \in \{0,1\}$ for logistic regression. Instead, we directly learn $p(y | \mathbf{x})$ with logistic regression, and then return the prediction $\hat{y}$ that has higher probability.
To limit the range of the learned probabilities within $[0,1]$, we use a *sigmoid* transformation. 
```math
p(y=1 | \mathbf{x})=σ(w_0+w_1x_1+w_2x_2+..+w_𝑛x_𝑛)
```
where $w_0$ represents the bias term. To take the bias term into account, we need to add a column of $1$s to the input matrix $$\mathbf{X}$$, resulting in the new dimension `(samples, features+1)`.

In this part you need to implement the sigmoid function. The sigmoid function takes in a scalar $z$ and is equal to $\text{sigmoid}(z) = 1/(1+ \exp(-z))$. To implement it, we can more generally assume it inputs a vector $\mathbf{z} = (z_1, z_2, \ldots, z_n)$ and outputs the sigmoid function applied to each element of this $\mathbf{z}$ (elementwise). Namely, $\text{sigmoid}(\mathbf{z}) = (1/(1+ \exp(-z_1)), 1/(1+ \exp(-z_2)), \ldots, 1/(1+ \exp(-z_n)))$. You can either implement this with a for loop, or use the fact that we can do elementwise operations on vectors. The `exp` function can be applied elementwise using `exp.(-z)`. And as before, elementwise addition is .+ and elementwise division is ./. If you find bugs using elementwise operations, then start with the straightforward for loop approach, get that working, and then experiment with elementwise operations.  
"

# ╔═╡ 0005ee8a-fcef-4c4e-8aa5-6140b06ed4ef
begin
	struct LogisticRegressor <: AbstractModel
		model::LinearModel
		γ::Float64 # the probabilty threshold on the output class confidence
		is_poly::Bool
	end
	
	LogisticRegressor(in, out=1; γ=0.5, is_poly=false) = if is_poly
		in = in - 1
		LogisticRegressor(LinearModel(in+1, out), γ, is_poly) # (feture size + 1 for bias term)  × output size
	else 
		LogisticRegressor(LinearModel(in+1, out), γ, is_poly) # (feture size + 1 for bias term)  × output size
	end
	Base.copy(lr::LogisticRegressor) = LogisticRegressor(copy(lr.model),lr.γ,lr.is_poly)
	get_linear_model(lr::LogisticRegressor) = lr.model
end;

# ╔═╡ 51599971-4638-4787-9c13-75afa0d34285
# Add a column of 1 to X to count for the bias term. Start with an "else" statement. 
function get_features(m::LogisticRegressor, X::AbstractMatrix)
	d = size(X, 2)
	_X = ones(size(X,1), d+1)
	_X[:, 1:d] = X
	X = _X
end;

# ╔═╡ 8847c374-c9f4-470d-80be-2946f2af9661
function sigmoid(z)
	z
	#### BEGIN SOLUTION
	z=1 ./ (1 .+exp.(-z))

	#### END SOLUTION
end;

# ╔═╡ 8745fec1-47c8-428b-9ea4-1e6828618830
begin
	__check_logit_reg = let
		rng = Random.MersenneTwister(1)
		_X = rand(rng, 3, 3)
		X = sigmoid(_X)
		X_true = [0.5587358993498943 0.5019773105398053 0.7215004060928302; 0.5857727098994119 0.6197795961579493 0.7310398330188039; 0.5775458635048137 0.5525472988567002 0.562585578409889]
		all(X .≈ X_true)
		end
	HTML("<h2 id=dist> Q1: Logistic Regression $(_check_complete(__check_logit_reg))")
end


# ╔═╡ ded749bf-b9fa-4e2b-b15f-0693d820a9c3
md"""
Now, we will implement Polynomial Model which uses the linear model for learning on top of non-linear features. We apply a polynomial transformation to our data, like the previous assignment, but now do a higher degree of $p=3$. 

Recall from the last assignment to obtain a polynomial transformation to our data to create new polynomial features. For $d$ inputs with a polynomial of size $p$, the number of features is $m = {d+p \choose p}$, giving polynomial function 

```math
f(\mathbf{x})=\sum_{j=1}^{m} w_j \phi_j (\mathbf{x}) = \boldsymbol{\phi}(\mathbf{x})^\top\mathbf{w}
```
We simply apply this transformation to every data point $\mathbf{x}_i$ to get the new dataset $\{(\boldsymbol{\phi}(\mathbf{x}_i), y_i)\}$.

Implement the polynomial feature transformation by constructing $\Phi$ with $p = 3$ degrees in the function ```get_features```.

"""

# ╔═╡ aa1dfa87-a844-4f82-9a30-008f15f88112
begin
	struct Polynomial3Model <: AbstractModel 
		model::LogisticRegressor
		ignore_first::Bool
	end
	
	Polynomial3Model(in, out=1; ignore_first=false) =
		Polynomial3Model(LogisticRegressor(1 + in + Int(in*(in+1)/2) + Int(floor((in*(in+1)/2)*(in+2)/3.0)), out, is_poly=true), ignore_first)

	Base.copy(lm::Polynomial3Model) = Polynomial3Model(copy(lm.model), lm.ignore_first)
	get_linear_model(lm::Polynomial3Model) = lm.model.model
	
end

# ╔═╡ 0ba5f9c8-5677-40e9-811b-25546e0df207
function get_features(pm::Polynomial3Model, _X::AbstractMatrix)
	# If _X already has a bias remove it.
	X = if pm.ignore_first
		_X[:, 2:end]
	else
		_X
	end
	
	m = size(X, 2)
	N = size(X, 1)
	num_features = 1 + # Bias bit
				   m + # p = 1
				   Int(m*(m+1)/2) + # combinations (i.e. x_i*x_j)
				   Int(floor(Int(m*(m+1)/2) * (m+2)/3))  # combinations (i.e. x_i*x_j*x_k)
	
	Φ = zeros(N, num_features)
	# Construct Φ
	#### BEGIN SOLUTION
	ind=1
	Φ[:,ind] .=1
	ind+=1
	Φ[:,ind:(ind+m-1)] .= X
	ind+=m

	for i in 1:m
		for j in i:m
			Φ[:,ind] .= X[:,i] .* X[:,j]
			ind+=1
		end
		for l in 1:m
			for f in l:m
				Φ[:,ind] .=X[:,i] .* X[:,l] .* X[:,f]
				ind+=1
			end
		end
	end
	

	#### END SOLUTION
	
	Φ
end;

# ╔═╡ 50cb6e7f-3341-47b8-b720-d069f03f1be2
function predict(lr::LogisticRegressor, X::AbstractMatrix)
	if lr.is_poly
		Ŷ = sigmoid(predict(lr.model, X))
	else
		Ŷ = sigmoid(predict(lr.model, get_features(lr, X))) 
	end
	pred = zeros(size(Ŷ))
	for i in 1:length(Ŷ)
		if Ŷ[i] >= lr.γ
			pred[i] = 1.0
		else
			pred[i] = 0.0
		end
	end
	pred
end;

# ╔═╡ c59cf592-a893-4ffa-b247-51d94c7cdb1a
begin
		
	__check_Poly2_logit_reg = let
		pm = Polynomial3Model(2, 1)
		rng = Random.MersenneTwister(1)
		X = rand(rng, 3, 2)
		Φ = get_features(pm, X)
		Φ_true = [1.0 0.23603334566204692 0.00790928339056074 0.05571174026441932 0.0018668546204633095 6.25567637522e-5 0.013149828447265864 0.00044063994193260575 1.4765482242222028e-5 4.947791725125075e-7; 1.0 0.34651701419196046 0.4886128300795012 0.12007404112451132 0.16931265897503248 0.2387424977182995 0.04160769821242834 0.058669717052929886 0.08272833747007607 0.11665264747038717; 1.0 0.3127069683360675 0.21096820215853596 0.09778564804593431 0.06597122691230639 0.04450758232200489 0.03057825354722182 0.020629662365158116 0.013917831135882103 0.00938968462489641]
		check_1 = check_elements(Φ_true, Φ) #all(Φ .≈ Φ_true)
		pm = Polynomial3Model(2, 1; ignore_first=true)
		X_bias = ones(size(X, 1), size(X, 2) + 1)
		X_bias[:, 2:end] .= X
		Φ = get_features(pm, X_bias)
		check_2 = check_elements(Φ_true, Φ)#all(Φ .≈ Φ_true)
		check_3 = (size(Φ)==size(Φ_true))
		check_1 && check_2 && check_3
	end
	
	HTML("<h2 id=poly> (b) Polynomial Features $(_check_complete(__check_Poly2_logit_reg))</h4>")
end

# ╔═╡ 0608c93d-2f82-470c-8d9f-ca79af6b2612
predict(lr::Polynomial3Model, X) = predict(lr.model, get_features(lr, X))

# ╔═╡ d9935cc8-ec24-47e9-b39a-92c21377a161
struct MiniBatchGD
	b::Int
end

# ╔═╡ e7712bd3-ea7e-4f4a-9efc-041b4b2be987
begin
	"""
		RandomModel
	
	Predicts `w*x` where `w` is sampled from a normal distribution.
	"""
	struct RandomModel <: AbstractModel # random weights
		W::Matrix{Float64}
		γ::Float64 # Threshold on binary classification confidence
	end
	RandomModel(in, out) = RandomModel(randn(in, out), 0.5)
	# predict(logit::RandomModel, X::AbstractMatrix) = sigmoid(X*logit.W) .>= Array(logit.γ, length(X*logit.W), 1) ? 1.0 : 0.0
	Base.copy(logit::RandomModel) = RandomModel(randn(size(logit.W)...), logit.γ)
	train!(::MiniBatchGD, model::RandomModel, lossfunc, opt, X, Y, num_epochs) = 
		nothing
end;

# ╔═╡ d77fe746-6fca-4a9e-97ac-0066db0ed2ca
function predict(logit::RandomModel, X::AbstractMatrix)
	Ŷ = sigmoid(X*logit.W) 
	pred = zeros(size(Ŷ))
	for i in 1:length(Ŷ)
		if Ŷ[i] >= logit.γ
			pred[i] = 1.0
		else
			pred[i] = 0.0
		end
	end
	pred
end;

# ╔═╡ 5714c84f-1653-4c4a-a2e4-003d8560484a
md"""
In this notebook, we will be focusing on minibatch gradient descent. Below you need to (re)implement the function `epoch!`. You can just use your code from Assignment 3 on `MBGD`. **We are not grading this section again, since it is the same code as before. But, if it is incorrect, it will give wrong results in later sections, resulting in deductions there. Please ensure this code is correct.** This function should go through the data set in mini-batches of size `mbgd.b`. Remember to randomize how you go through the data **and** that you are using the correct targets for the data passed to the learning update. In this implementation, you will use 

```julia
update!(model, lossfunc, opt, X_batch, Y_batch)
```

to update your model. You randomize and divide the dataset into batches and call the update function for each batch. These update functions are defined in the section on [optimizers](#opt).

"""

# ╔═╡ 9d96ede3-533e-42f7-ada1-6e71980bc6c2
function epoch!(mbgd::MiniBatchGD, model::AbstractModel, lossfunc, opt, X, Y)
	epoch!(mbgd, get_linear_model(model), lossfunc, opt, get_features(lp.model, X), Y)
end;

# ╔═╡ 6ff92fca-6d66-4f27-8e09-11a3887e66ba
function train!(mbgd::MiniBatchGD, model::AbstractModel, lossfunc, opt, X, Y, num_epochs)
	train!(mbgd, get_linear_model(model), lossfunc, opt, get_features(model, X), Y, num_epochs)
end;

# ╔═╡ 7e777dba-b389-4549-a93a-9b0394646c57
abstract type LossFunction end

# ╔═╡ f380a361-2960-471b-b29a-3bd1fe06252b
md"""
#### (d) Cross-entropy
"""

# ╔═╡ 6d2d24da-9f3f-43df-9243-fc17f85e0b01
md"""
The cross-entropy loss function for the whole dataset is

```math
c(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n c_i(\mathbf{w})
```
for cross-entropy loss $c_i$ on datapoint $i$
```math
c_i(\mathbf{w}) = −y_i  \ln  σ(\mathbf{x}_i^T \mathbf{w}) - (1 − y_i) \ln(1 − \sigma(\mathbf{x}_i^T \mathbf{w}))
```

You should be using the sigmoid function $\sigma$ that you implemented above. When computing the loss for a minibatch $\{i_1, i_2, \ldots, i_b\}$, we sum losses $c_i$ for those points

```math
\frac{1}{b}\sum_{j=1}^b c_{i_j}(\mathbf{w})
```
You will compute the loss for a given batch of data, inputted into `loss` as `X` and `Y`. This data could be the entire dataset, or a minibatch. 

"""

# ╔═╡ 4f43373d-42ee-4269-9862-f53695351ea3
struct CrossEntropy <: LossFunction end

# ╔═╡ ada800ba-25e2-4544-a297-c42d8b36a9ff
function loss(lm::AbstractModel, ce::CrossEntropy, X , Y)
	θ = predict(lm, X) # θ = Xw
	loss = 0.0
	#### BEGIN SOLUTION

	for i in 1:size(X,1)
		y=sigmoid(θ[i])
		loss+=-(y*log(y) + (1-y) * log(1-y))
	end
	return (1/size(X,1))*loss
	#### END SOLUTION
end;

# ╔═╡ 7bea0c90-077f-4eca-b880-02d1289244f3
md"""
#### (e) Gradient of Cross-Entropy
"""

# ╔═╡ 4ea14063-99ca-4caf-a862-fbf9590c68a2
md"""
You will implement the gradient of the cross-entropy loss function in the `gradient` function, returning a matrix of the same size of `lm.w` using the following formula:
```math
\nabla c_i(\mathbf{w}) = (σ(\mathbf{x}_i^T \mathbf{w}) − y_i)\mathbf{x}_i
```

"""

# ╔═╡ 299116ea-66f3-4e52-ab0f-594249b9dd23
function gradient(lm::AbstractModel, ce::CrossEntropy, X::Matrix, Y::Vector)
	∇w = zero(lm.w) # gradients should be the size of the weights
	
	#### BEGIN SOLUTION
	x=predict(lm,x)
	hx=sigmoid.(x)
	s=size(X,1)
	err=hx .- Y
	∇w = (1/s) .* (X' * error)

	#### END SOLUTION

	@assert size(∇w) == size(lm.w)
	∇w
end;

# ╔═╡ 36c1f5c8-ac43-41ea-9100-8f85c1ee3708
abstract type Optimizer end

# ╔═╡ 159cecd9-de77-4586-9479-383661bf3397
begin
	struct _LR <: Optimizer end
	struct _LF <: LossFunction end
	function gradient(lm::LinearModel, lf::_LF, X::Matrix, Y::Vector)
		sum(X, dims=1)
	end
	function update!(lm::LinearModel, 
		 			 lf::_LF, 
		 			 opt::_LR, 
		 			 x::Matrix,
		 			 y::Vector)
		
		ϕ = get_features(lm, x)
		
		Δw = gradient(lm, lf, ϕ, y)[1, :]
		lm.w .-= Δw
	end
end;

# ╔═╡ a3387a7e-436c-4724-aa29-92e78ea3a89f
begin
	_X = [4 3 4 1; 1 0 5 1; 1 5 6 1; 4 4 7 1; 2 4 8 1]
	__check_cegrad = all(gradient(LinearModel(4, 1), CrossEntropy(), _X, [1.0,0.0,0.0,1.0,1.0]) .≈ [-0.8; -0.6000000000000001; -0.8; -0.1])
	__check_celoss = loss(LinearModel(4, 1), CrossEntropy(), _X, [1.0,0.0,0.0,1.0,1.0]) .≈ 0.6931471805599454
	
	__check_CrossEntropy = __check_celoss && __check_cegrad
	
md"""
For this notebook we only use the cross-entropy, but we still use the abstract type LossFunction as a standard abstract type for all losses. Below you will need to implement the `loss` $(_check_complete(__check_celoss)) function and the `gradient` $(_check_complete(__check_cegrad)) function for Cross_Entropy.
"""
end

# ╔═╡ a17e5acd-d78d-4fab-9ab2-f01bd888339d
HTML("<h3 id=lossfunc> Loss Functions $(_check_complete(__check_CrossEntropy)) </h3>")

# ╔═╡ 77cda37c-c560-42d8-85b1-7f026d09edfe
md"""
RMSprop is another adaptive learning rate that uses a different learning rate for every parameter $w_j$. Instead of taking cumulative sum of squared gradients as in AdaGrad, we take the exponential moving average of these gradients. The motivation for doing so is to allow the gradient to stay larger for longer, if needed. The stepsize for Adagrad can decay quite quickly, due to the accumulation of squared gradients. 

To implement RMSprop, we use the following equations:

```math
\begin{align}
\bar{\mathbf{g}}_{t+1} &= \beta \bar{\mathbf{g}}_{t} + (1-\beta) \mathbf{g}_t^2 \\
\mathbf{w}_{t+1} &= \mathbf{w}_{t} - \frac{\eta}{\sqrt{\bar{\mathbf{g}}_{t+1} + \epsilon}}  \mathbf{g}_t
\end{align}
```
where $\mathbf{g}_t$ is the gradient at time step $t$ and the addition, squaring, multiplication and division are all elementwise. The coefficient $\beta$ represents the degree of weighting decrease, a constant smoothing factor between $0$ and $1$. A higher $\beta$ discounts older observations faster.

Implement ```RMSprop``` below.
"""

# ╔═╡ 1fe7084a-9cf9-48a4-9e60-b712365eded9
begin
	mutable struct RMSprop <: Optimizer
		η::Float64 # step size
		β::Float64 # The significance coefficient on the most recent data points
		gbar::Matrix{Float64} # exponential decaying average
		ϵ::Float64 # 
	end
	
	RMSprop(η) = RMSprop(η, 0.9, zeros(1, 1), 1e-5)
	RMSprop(η, lm::LinearModel) = RMSprop(η, 0.9, zero(lm.w), 1e-5)
	RMSprop(η, model::AbstractModel) = RMSprop(η, get_linear_model(model))
	Base.copy(rmsprop::RMSprop) = RMSprop(rmsprop.η, rmsprop.β, zero(rmsprop.gbar), rmsprop.ϵ)
end

# ╔═╡ c2710a60-ebe1-4d01-b6d1-0d6fe45723f9
function update!(lm::LinearModel, 
				 lf::LossFunction,
				 opt::RMSprop,
				 x::Matrix,
				 y::Vector)

	g = gradient(lm, lf, x, y)
	if size(g) !== size(opt.gbar) # need to make sure this is of the right shape.
		opt.gbar = zero(g)
	end
	
	# update opt.gbar and lm.w	
	#### BEGIN SOLUTION
	opt.gbar .= (opt.β * opt.gbar) .+ (1 - opt.β) .* (g .^2)
	div=sqrt.(opt.gbar .+ opt.ϵ)
	lm.w .= lm.w .- (opt.η ./ div) .* g

	#### END SOLUTION	
end;

# ╔═╡ 69cf84e2-0aba-4595-8cb0-c082dbccdbe2
function epoch!(mbgd::MiniBatchGD, model::LinearModel, lossfunc, opt, X, Y)
	
	#### BEGIN SOLUTION
	
	b=mbgd.b
	ran=randperm(size(X,1))
	s_x=X[ran,:]
	rem=X[ran]
	s=size(s_x,1)
	for i in (1:b:s)
		X_batch=s_x[i:i+b-1,:]
		Y_batch=rem[i:i+b-1]
		update!(model,lossfunc,opt,X_batch,Y_batch)
	end
	
	

	
	#### END SOLUTION
end;

# ╔═╡ acf1b36c-0412-452c-ab4d-a388f84fd1fb
begin
	__check_MBGD = let

		lm = LinearModel(3, 1)
		opt = _LR()
		lf = _LF()
		X = ones(10, 3)
		Y = collect(0.0:0.1:0.9)
		mbgd = MiniBatchGD(5)
		epoch!(mbgd, lm, lf, opt, X, Y)
		all(lm.w .== -10.0)
	end
	str = "<h2 id=graddescent> (c) Mini-batch Gradient Descent $(_check_complete(__check_MBGD)) </h2>"
	HTML(str)
end

# ╔═╡ 2782903e-1d2e-47de-9109-acff4595de42
function train!(mbgd::MiniBatchGD, model::LinearModel, lossfunc, opt, X, Y, num_epochs)
	ℒ = zeros(num_epochs + 1)
	ℒ[1] = loss(model, lossfunc, X, Y)
	for i in 1:num_epochs
		epoch!(mbgd, model, lossfunc, opt, X, Y)
		ℒ[i+1] = loss(model, lossfunc, X, Y)
	end
	ℒ
end;

# ╔═╡ 8dfd4734-5648-42f2-b93f-be304b4b1f27
begin
	 __check_RMSprop_v, __check_RMSprop_W = let
		lm = LinearModel(2, 1)
		opt = RMSprop(0.1, lm)
		X = [0.1 0.5; 
			 0.5 0.0; 
			 1.0 0.2]
		Y = [1, 0, 1]
		update!(lm, CrossEntropy(), opt, X, Y)
		true_G = [0.0009999999999999996; 0.0013611111111111105]
		true_W = [0.31465838776377636; 0.31507247500483543]
		all(opt.gbar .≈ true_G), all(lm.w .≈ true_W)
	end
	
	__check_RMSprop = __check_RMSprop_v && __check_RMSprop_W
	
md"""
#### (f) RMSprop $(_check_complete(__check_RMSprop))

	
"""
end

# ╔═╡ af8acfdf-32bd-43c1-82d0-99008ee4cb3e
HTML("<h3 id=opt> Optimizers $(_check_complete(__check_RMSprop)) </h3>")

# ╔═╡ 3738f45d-38e5-415f-a4e6-f8922df84d09
md"""
Below you will need to implement an optimizer:

- RMSprop $(_check_complete(__check_RMSprop))
"""

# ╔═╡ fa610de0-f8c7-4c48-88d8-f5398ea75ae2
md"""
# Evaluating models

In the following section, we provide a few helper functions and structs to make evaluating methods straightforward. The abstract type `LearningProblem` with child `GDLearningProblem` is used to construct a learning problem. Once again, we introduce an abstract type even though we only have one child, but provide a design that does not constrain the types of learners that are possible. This struct contain all the information needed to `train!` a model for gradient descent. We also provide the `run` and `run!` functions. These will apply the feature transform (if there is one) and train the model. `run` does this with a copy of the learning problem, while `run!` does this inplace. 

"""

# ╔═╡ d695b118-6d0d-401d-990f-85ba467cc53e
abstract type LearningProblem end

# ╔═╡ 6edc243e-59ac-4c6f-b507-80d3ec13bc21
"""
	GDLearningProblem

This is a struct for keeping a the necessary gradient descent learning setting components together.
"""
struct GDLearningProblem{M<:AbstractModel, O<:Optimizer, LF<:LossFunction} <: LearningProblem
	gd::MiniBatchGD
	model::M
	opt::O
	loss::LF
end;

# ╔═╡ 3bdde6cf-3b68-46d3-bf76-d68c20b661e9
Base.copy(lp::GDLearningProblem) = 
	GDLearningProblem(lp.gd, copy(lp.model), copy(lp.opt), lp.loss)

# ╔═╡ 7905f581-1593-4e06-8aaf-faec05c3b306
function run!(lp::GDLearningProblem, X, Y, num_epochs)
	update_transform!(lp.model, X, Y)
	train!(lp.gd, lp.model, lp.loss, lp.opt, X, Y, num_epochs)
end;

# ╔═╡ 69b96fc3-dc9c-44de-bc7f-12bb8aba85d1
function run(lp::LearningProblem, args...)
	cp_lp = copy(lp)
	ℒ = run!(cp_lp, args...)
	return cp_lp, ℒ
end;

# ╔═╡ 1cdb6679-c18f-46f7-8f23-9ed6e138a7a9
md"""
### Accuracy and Misclassification Error
The Accuracy of a model is the percent correct on the given (testing) data, namely the percent of points where we predicted the correct class. The misclassification error is 100 minus this number, shows the percent incorrect. The misclassification error on the test set is an estimate of the expected 0-1 cost of the classifier, given as a percent rather than a percentage between 0 and 1. 
"""

# ╔═╡ 89cc730e-ab66-4f87-827c-87539ac1f54a
function get_accuracy(Y, Ŷ)
    correct = 0
    # count number of correct predictions
    correct = sum(Y .== Ŷ)
    # return percent correct
    return (correct / Float64(length(Y))) * 100.0
end;

# ╔═╡ 045b8be8-58c6-497b-baac-8af41de76b1e
function get_misclassification_error(Y, Ŷ)
    return (100 - get_accuracy(Y, Ŷ))
end;

# ╔═╡ eef918a9-b8af-4d41-85b1-bebf1c7889cc
HTML("<h4 id=cv> Run Experiment </h2>")

# ╔═╡ fd75ff49-b5de-48dc-ae89-06bf855d81b2
md"""

Below are the helper functions for running an experiment.

"""

# ╔═╡ d339a276-296a-4378-82ae-fe498e9b5181
"""
	run_experiment(lp, X, Y, num_epochs, runs; train_size)

Using `train!` do `runs` experiments with the same train and test split (which is made by `random_dataset_split`). This will create a copy of the learning problem and use this new copy to train. It will return the estimate of the error.
"""
function run_experiment(lp::LearningProblem, 
						data,	 
						num_epochs,
						runs)
	err = zeros(runs)

	X, Y = data.X, data.Y
	train_size=18000
	test_size = 3000
	
	for i in 1:runs		
		
		rp = randperm(length(Y))
		train_idx = rp[1:train_size]
		test_idx = rp[train_size+1:train_size+test_size]
		train_data = (X[train_idx, :], Y[train_idx]) 
		test_data = (X[test_idx, :], Y[test_idx])
		
		# train
		cp_lp, train_loss = run(lp, train_data[1], train_data[2], num_epochs)
		
		# test
		Ŷ = predict(cp_lp.model, test_data[1])
		err[i] = get_misclassification_error(test_data[2], Ŷ)
	end

	err
end;

# ╔═╡ 58e626f1-32fb-465a-839e-1f413411c6f3
md"
# Experiments

In this section, we will run an experiment on the algorithms we implemented above. We provide the data in the `Data` section, and then follow the experiment and its description. You will need to analyze and understand the experiment for the written portion of this assignment.
"

# ╔═╡ 5ec88a5a-71e2-40c1-9913-98ced174341a
md"""
## The Dataset

We use a the physics dataset, from the UCI repository. We normalize the columns using min-max scaling. A description of the dataset is given below, given by the group that released the dataset. 
"""

# ╔═╡ d2c516c0-f5e5-4476-b7d6-89862f6f2472
function unit_normalize_columns!(df::DataFrame)
	for name in names(df)
		mn, mx = minimum(df[!, name]), maximum(df[!, name])
		df[!, name] .= (df[!, name] .- mn) ./ (mx - mn)
	end
	df
end;

# ╔═╡ 90f34d85-3fdc-4e2a-ada4-085154103c6b
physiscs_data = let
	data = CSV.read("data/susysubset.csv", DataFrame, delim=',', ignorerepeated=true)[:, 1:end]
	data[!, 1:end-1] = unit_normalize_columns!(data[:, 1:end-1])
	data
end;

# ╔═╡ eac4fb9d-126b-4ba8-9078-105638416de2
md"""
Here is a description of the physics dataset in case you are interested.

>> The data has been produced using Monte Carlo simulations and contains events with two leptons (electrons or muons). In high energy physics experiments, such as the ATLAS and CMS detectors at the CERN LHC, one major hope is the discovery of new particles. To accomplish this task, physicists attempt to sift through data events and classify them as either a signal of some new physics process or particle, or instead a background event from understood Standard Model processes. Unfortunately we will never know for sure what underlying physical process happened (the only information to which we have access are the final state particles). However, we can attempt to define parts of phase space that will have a high percentage of signal events. Typically this is done by using a series of simple requirements on the kinematic quantities of the final state particles, for example having one or more leptons with large amounts of momentum that is transverse to the beam line ( pT ). Here instead we will use logistic regression in order to attempt to find out the relative probability that an event is from a signal or a background event and rather than using the kinematic quantities of final state particles directly we will use the output of our logistic regression to define a part of phase space that is enriched in signal events. The dataset we are using has the value of 18 kinematic variables ("features") of the event. The first 8 features are direct measurements of final state particles, in this case the  pT , pseudo-rapidity ( η ), and azimuthal angle ( ϕ ) of two leptons in the event and the amount of missing transverse momentum (MET) together with its azimuthal angle. The last ten features are functions of the first 8 features; these are high-level features derived by physicists to help discriminate between the two classes. You can think of them as physicists attempt to use non-linear functions to classify signal and background events and they have been developed with a lot of deep thinking on the part of physicist. There is however, an interest in using deep learning methods to obviate the need for physicists to manually develop such features. Benchmark results using Bayesian Decision Trees from a standard physics package and 5-layer neural networks and the dropout algorithm are presented in the original paper to compare the ability of deep-learning to bypass the need of using such high level features. We will also explore this topic in later notebooks. The dataset consists of 5 million events, the first 4,500,000 of which we will use for training the model and the last 500,000 examples will be used as a test set.

"""

# ╔═╡ 14b329fb-8053-4148-8d24-4458e592e7e3
md"""
## Plotting our results

The `plot_results` function produces two plots. The left plot is a box plot over the errors, the right plot is a bar graph displaying average errors with standard error bars. This function will be used for all the experiments, and you should use this to finish your written experiments.

"""


# ╔═╡ eebf5285-2336-4c07-a4fd-b1fd841dee52
function plot_results(algs, errs; vert=false)
	stderr(x) = sqrt(var(x)/length(x))
	
	plt1 = boxplot(reshape(algs, 1, :),
				   errs,
				   legend=false, ylabel="Misclassification error",
				   pallette=:seaborn_colorblind)
	
	plt2 = bar(reshape(algs, 1, :),
			   reshape(mean.(errs), 1, :),
			   yerr=reshape(stderr.(errs), 1, :),
			   legend=false,
			   pallette=:seaborn_colorblind,
			   ylabel=vert ? "Misclassification error" : "")
	
	if vert
		plot(plt1, plt2, layout=(2, 1), size=(600, 600))
	else
		plot(plt1, plt2)
	end
end;

# ╔═╡ 9ed07108-2ed0-430f-ab97-6f51297c5361
md"""
## (g) Evaluating the Classifiers

We will compare different classifiers on the [Physics dataset](). 

To run this experiment click $(@bind __run_class PlutoUI.CheckBox())

**You can get the misclassification error to report from the plot or from the terminal where your ran this notebook**.

Note that it might take a bit of time to run this experiment (even a few minutes), so be a bit patient waiting for the graph to appear right below this text.
"""

# ╔═╡ d686c8ca-cb29-4f7c-8872-a907173b156c
begin
	if __run_class
		algs = ["Random", "Logit", "PolyLogit"]
		classification_problems = [
			GDLearningProblem(
				MiniBatchGD(200),
				RandomModel(8, 1),
				RMSprop(0.01),
				CrossEntropy()),
			GDLearningProblem(
				MiniBatchGD(200),
				LogisticRegressor(8, 1),
				RMSprop(0.01),
				CrossEntropy()),
			GDLearningProblem(
				MiniBatchGD(200),
				Polynomial3Model(8, 1),
				RMSprop(0.01),
				CrossEntropy())
			];
		
		misclass_errs = let
			Random.seed!(2)
			data = (X=Matrix(physiscs_data[:, 1:end-1]), Y=physiscs_data[:, end])
			@show size(data.X)
			errs = Vector{Float64}[]
			
			for (idx, prblms) in enumerate(classification_problems)

				err = run_experiment(prblms, data, 200, 10)
				push!(errs, err)
			end
			errs
		end

		mean_error_Random = mean(misclass_errs[1])
		mean_error_Logit = mean(misclass_errs[2])
		mean_error_PolyLogit = mean(misclass_errs[3])
		
		println("Misclassification error on the test set for Random model is $mean_error_Random.")
		
		println("Misclassification error on the test set for Logistic Regression model is $mean_error_Logit.")
		
		println("Misclassification error on the test set for Polynomial Logistic Regression model is $mean_error_PolyLogit.")
		
		plot_results(algs, misclass_errs)
	end

end

# ╔═╡ 84ae70a8-3ce3-468c-81b7-ba423745a32a


# ╔═╡ 14fa89f8-d034-4286-bdb2-2c11190e17d0
md"""
## Q2: Hypothesis Testing
"""

# ╔═╡ 26466a2e-5554-407f-8729-e2b841f10a7e
md"""

In this question, you will use the paired t-test to compare the performance of two models. You will compare the two models from above (logistic regression and polynomial logistic regression) both using RMSprop for optimization. You hypothesize that polynomial logistic regression is better than logistic regression and you want to run a one-tailed t-test to see if this is true. 

The paired t-test is designed for continuous errors, rather than 0-1 errors. For 0-1 errors, and one test set, we often use McNemar's test. However, there is another alternative. We can run the algorithm on multiple random training and testing splits (say 10) to get 10 estimates of misclassification error (average 0-1 error on a test set, as a percent). The standard deviation and average of these 10 errors provides a useful insight into the quality of the model we would get for this problem. We therefore run each algorithm 10 times, on random training and test splits. We control the splits so that both algorithms train and test on the same split, to have paired samples for the 10 samples of error. We then conduct a paired t-test on these 10 numbers. 

The misclassification error estimate above, in fact, is the average of these 10 numbers. It looks like polynomial logistic regression is better than logistic regression. Now we will examine whether we can say this with confidence (that the difference is statistically significant). 

"""

# ╔═╡ 1189ddd5-9295-4e1c-a50e-11efed56d35b
md"""
### (a) Defining the Null Hypothesis
"""

# ╔═╡ 9e33cb14-44b6-4335-a899-7d51a9829346
md"""
Define the null hypothesis and the alternative hypothesis. Assume $\mu_1$ to be the true expected misclassification error for Logistic Regression and $\mu_2$ to be the true expected misclassification error for Polynomial Logistic Regression. This is not code that is run, rather it is simply your written answer to your question. For marking purposes, it is easier for us if you to write this description here. Note that to write these answers, you have to put them in comments because they are not valid Julia code. 
"""

# ╔═╡ 11098780-5235-40a1-9477-091ce68420a9
# discussion should go here
#### BEGIN SOLUTION
#Ho: Polynomial logistic regression is better than logistic regression (µ1>µ2)
#Ha: Polynomial logistic regression is not better than logistic regression (µ1<µ2)
#### END SOLUTION

# ╔═╡ 91636720-2e59-47ab-87ff-a50e9d817d18
md"""
To run the one-tailed t-test, you need to compute the p-value. To do this implement the ```getPValue``` method, which returns the p-value for the one-tailed paired t-test. It looks at the positive part of the tail: we consider the difference between the baseline and learner, where if it is a larger positive number that the error for the learner is much lower than the baseline. This test therefore checks if the learner (polynomial logistic regression) is statistically significantly better than the baseline (logistic regression). 

"""

# ╔═╡ 2cd21c3b-c826-40d6-9bdf-5e808c23a7d3
# helper function to get the positive tail p-value using t-distribution
function pValueTDistPositiveTail(t::Float64, dof::Int64)
	1 - cdf(TDist(dof), t)
end;

# ╔═╡ c71c562b-bb4f-4688-a451-daf2040ede62
function tDistPValue(baseline_error::AbstractVector{Float64}, 		           learner_error::AbstractVector{Float64})
	# Computes the p-value using one-tailed t-test
	@assert size(learner_error) == size(baseline_error)
	m = size(learner_error, 1) # the number of test samples
	dof = m - 1
	t = 0.0
	#### BEGIN SOLUTION
	d=baseline_error .- learner_error
	mean_value=mean(d)
	s_error=std(d)/sqrt(m)
	t=mean_value/s_error

	#### END SOLUTION
	pValueTDistPositiveTail(t, dof)
end;

# ╔═╡ f8ff4eca-1d38-45dc-9d85-5e0de774aa4a
begin
	__check_q2c = let
		e1 = [30.5, 44.3, 46.8, 50.9]
		e2 = [50.5, 49.3, 51.8, 55.5]
		pval = tDistPValue(e1, e2)
		pval == 0.946808781297146
	end
	HTML("<h2 id=t_test> (b) Implementing the t-test $(_check_complete(__check_q2c))")
end

# ╔═╡ a9d3c6c3-1cb7-4417-ba6a-54998841c87d
let
	q1_a_check = _check_complete(__check_logit_reg)
	q1_b_check = _check_complete(__check_Poly2_logit_reg)
	q1_c_check = _check_complete(__check_MBGD)
	q1_d_check = _check_complete(__check_celoss)
	q1_e_check = _check_complete(__check_cegrad)
	q1_f_check = _check_complete(__check_RMSprop)

md"""
# Preamble
	
In this assignment, we will implement:
	
- Q1(a) [Logistic Regression: sigmoid function](#logit) $(q1_a_check)
- Q1(b) [Polynomial Logistic Regression](#graddescent) $(q1_b_check)
- Q1(c) [Mini-batch Gradient Descent](#gd) use the old code in Assignment 3 $(q1_c_check) 
- Q1(d) [Loss Function](#lossfunc) cross entropy $(q1_d_check)
- Q1(e) [Gradient of Loss Function](#gradlossfunc) gradient of cross entropy $(q1_e_check)
- Q1(f) [Optimizer](#opt): adapive stepsize RMSprop $(q1_f_check)
- Q2(a) Hypothesis-testing: Define Null hypothesis and alternative hypothesis
- Q2(c) [Running the t-test](#t_test): get the pvalue and run the t-test $(__check_q2c)
"""
end

# ╔═╡ efe707a1-41c7-439d-9e57-dfea374f355d
md"""
### (c) Running the t-test

Next, you will run the t_test. Report the p-value. Would you be able to reject the null hypothesis with a significance threshold of 0.01? Include both the p-value as well as this discussion in the comments right below this text.
"""

# ╔═╡ 015bdde5-77c9-49d7-a8ba-c9c11f0033a9
# discussion should go here
#### BEGIN SOLUTION


#### END SOLUTION

# ╔═╡ 62a4c01b-434b-4a2b-a46b-78bee5136dad
function t_test(baseline_error::AbstractVector{Float64},
		learner_error::AbstractVector{Float64},
		learner_name::AbstractString, 
		baseline_name::AbstractString, 
		pvalueThreshold::Float64)

    pval = tDistPValue(baseline_error, learner_error)
	
    if pval < pvalueThreshold
        result = "rejected"
    else
        result = "not rejected"
	end
    println("With pvalue = $pval, the null hypothesis is $result under a pvalueThreshold of $pvalueThreshold")
end;

# ╔═╡ 07314336-5096-4fd3-a94a-70e41a7d3a6a
begin
	if __run_class
		baseline_error = misclass_errs[2]
		learner_error =  misclass_errs[3]
		
		baseline_name = "LogisticRegression"
		learner_name = "PolynomialLogisticRegression"
	
		pvalueThreshold = 0.01
	
		t_test(baseline_error, learner_error, learner_name, baseline_name, pvalueThreshold)
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.10.4"
DataFrames = "~1.3.4"
Distributions = "~0.25.66"
MultivariateStats = "~0.9.1"
PlutoUI = "~0.7.39"
StatsBase = "~0.33.20"
StatsPlots = "~0.15.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.4"
manifest_format = "2.0"
project_hash = "875844c39a3b7ab001051ecd333a98b00a15b1e6"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "aafa0665e3db0d3d0890cdc8191ea03dc279b042"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.66"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "037a1ca47e8a5989cc07d19729567bb71bfabd0c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.66.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "c8ab731c9127cd931c93221f65d6a1008dad7256"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.66.0+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "a7a97895780dab1085a97769316aa348830dc991"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.3"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ed47af35905b7cc8f1a522ca684b35a212269bd8"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.2.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "d19f9edd8c34760dca2de2b503f969d8700ed288"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.4"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "23e651bbb8d00e9971015d0dd306b780edbdb6b9"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.3"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "1a43be956d433b5d0321197150c2f94e16c0aaa0"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.16"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "361c2b088575b07946508f135ac556751240091c"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.17"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "d9ab10da9de748859a7780338e1d6566993d1f25"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

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

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "7008a3412d823e29d370ddc77411d593bd8a3d03"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "0e353ed734b1747fc20cd4cba0edd9ac027eff6a"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "05873db92e703f134649d88b8a164f3b7acb4d73"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

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

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "e7eac76a958f8664f2718508435d058168c7953d"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "22c5201127d7b243b9ee1de3b43c408879dff60f"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.3.0"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

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
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "23368a3313d12a2326ad0035f0db0c0966f438ef"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "66fe9eb253f910fe8cf161953880cfdaef01cdf0"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.1"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8d7530a38dbd2c397be7ddd01a424e4f411dcc41"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.2"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "0005d75f43ff23688914536c5e9d5ac94f8077f7"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.20"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "2b35ba790f1f823872dcf378a6d3c3b520092eac"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "ec47fb6069c57f1cee2f67541bf8f23415146de7"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.11"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─75441ce6-2137-4fcf-bba2-6ed67b9acb59
# ╠═dc004086-db6e-4813-841e-d427520402f7
# ╠═c1b9d7a3-5605-4478-b1a9-a9553ad38ed7
# ╠═c9797979-7d16-4ba4-8f82-c7ec33f6e517
# ╟─693a3933-c1c2-4249-8c03-f5151267222f
# ╟─bc0c5abe-1c2a-4e5c-bb0c-8d76b5e81133
# ╠═def97306-1703-42bc-bc09-da623c545e87
# ╟─14c30b50-157f-40bf-b0a6-0232aa2705c6
# ╠═4a196e82-fe94-49fe-ab87-d57e2d7fbd34
# ╟─a9d3c6c3-1cb7-4417-ba6a-54998841c87d
# ╠═7247e88f-e9c3-4561-ae67-71eb23ad066d
# ╠═a7aecd21-13f2-4cf2-8239-3a3d708602c9
# ╟─e3c3e903-e2c2-4de0-a3b2-56a27709e8c3
# ╟─a35944ae-7bbb-4f3c-bbe0-d2bbb57cd70b
# ╠═e7712bd3-ea7e-4f4a-9efc-041b4b2be987
# ╠═d77fe746-6fca-4a9e-97ac-0066db0ed2ca
# ╟─7b513344-1cad-4eef-9faf-e77ba176323e
# ╟─4f4029a2-c590-4bd3-a0db-d2380d4b4620
# ╟─dcfecc35-f25c-4856-8429-5c31d94d0a42
# ╠═d45dedf8-e46a-4ffb-ab45-f9d211d3a8ca
# ╠═7cd46d84-a74c-44dc-8339-68010924bc39
# ╟─8745fec1-47c8-428b-9ea4-1e6828618830
# ╟─fd4df2c6-fbfe-4089-a487-e3965a463ef3
# ╠═2d43a3ba-2a2c-4114-882f-5834d42e302a
# ╟─9c0491db-ad27-46fc-969c-c42a14cdadeb
# ╠═0005ee8a-fcef-4c4e-8aa5-6140b06ed4ef
# ╠═51599971-4638-4787-9c13-75afa0d34285
# ╠═50cb6e7f-3341-47b8-b720-d069f03f1be2
# ╠═8847c374-c9f4-470d-80be-2946f2af9661
# ╠═c59cf592-a893-4ffa-b247-51d94c7cdb1a
# ╟─ded749bf-b9fa-4e2b-b15f-0693d820a9c3
# ╠═aa1dfa87-a844-4f82-9a30-008f15f88112
# ╠═0608c93d-2f82-470c-8d9f-ca79af6b2612
# ╠═0ba5f9c8-5677-40e9-811b-25546e0df207
# ╠═acf1b36c-0412-452c-ab4d-a388f84fd1fb
# ╠═159cecd9-de77-4586-9479-383661bf3397
# ╠═d9935cc8-ec24-47e9-b39a-92c21377a161
# ╟─5714c84f-1653-4c4a-a2e4-003d8560484a
# ╠═69cf84e2-0aba-4595-8cb0-c082dbccdbe2
# ╠═9d96ede3-533e-42f7-ada1-6e71980bc6c2
# ╠═6ff92fca-6d66-4f27-8e09-11a3887e66ba
# ╠═2782903e-1d2e-47de-9109-acff4595de42
# ╟─a17e5acd-d78d-4fab-9ab2-f01bd888339d
# ╟─a3387a7e-436c-4724-aa29-92e78ea3a89f
# ╠═7e777dba-b389-4549-a93a-9b0394646c57
# ╠═f380a361-2960-471b-b29a-3bd1fe06252b
# ╟─6d2d24da-9f3f-43df-9243-fc17f85e0b01
# ╠═4f43373d-42ee-4269-9862-f53695351ea3
# ╠═ada800ba-25e2-4544-a297-c42d8b36a9ff
# ╠═7bea0c90-077f-4eca-b880-02d1289244f3
# ╟─4ea14063-99ca-4caf-a862-fbf9590c68a2
# ╠═299116ea-66f3-4e52-ab0f-594249b9dd23
# ╟─af8acfdf-32bd-43c1-82d0-99008ee4cb3e
# ╟─3738f45d-38e5-415f-a4e6-f8922df84d09
# ╠═36c1f5c8-ac43-41ea-9100-8f85c1ee3708
# ╟─8dfd4734-5648-42f2-b93f-be304b4b1f27
# ╟─77cda37c-c560-42d8-85b1-7f026d09edfe
# ╠═1fe7084a-9cf9-48a4-9e60-b712365eded9
# ╠═c2710a60-ebe1-4d01-b6d1-0d6fe45723f9
# ╠═fa610de0-f8c7-4c48-88d8-f5398ea75ae2
# ╠═d695b118-6d0d-401d-990f-85ba467cc53e
# ╠═6edc243e-59ac-4c6f-b507-80d3ec13bc21
# ╠═3bdde6cf-3b68-46d3-bf76-d68c20b661e9
# ╠═7905f581-1593-4e06-8aaf-faec05c3b306
# ╠═69b96fc3-dc9c-44de-bc7f-12bb8aba85d1
# ╟─1cdb6679-c18f-46f7-8f23-9ed6e138a7a9
# ╠═89cc730e-ab66-4f87-827c-87539ac1f54a
# ╠═045b8be8-58c6-497b-baac-8af41de76b1e
# ╟─eef918a9-b8af-4d41-85b1-bebf1c7889cc
# ╟─fd75ff49-b5de-48dc-ae89-06bf855d81b2
# ╠═d339a276-296a-4378-82ae-fe498e9b5181
# ╠═58e626f1-32fb-465a-839e-1f413411c6f3
# ╟─5ec88a5a-71e2-40c1-9913-98ced174341a
# ╠═d2c516c0-f5e5-4476-b7d6-89862f6f2472
# ╠═90f34d85-3fdc-4e2a-ada4-085154103c6b
# ╟─eac4fb9d-126b-4ba8-9078-105638416de2
# ╟─14b329fb-8053-4148-8d24-4458e592e7e3
# ╠═eebf5285-2336-4c07-a4fd-b1fd841dee52
# ╟─9ed07108-2ed0-430f-ab97-6f51297c5361
# ╠═d686c8ca-cb29-4f7c-8872-a907173b156c
# ╟─84ae70a8-3ce3-468c-81b7-ba423745a32a
# ╟─14fa89f8-d034-4286-bdb2-2c11190e17d0
# ╠═26466a2e-5554-407f-8729-e2b841f10a7e
# ╠═1189ddd5-9295-4e1c-a50e-11efed56d35b
# ╟─9e33cb14-44b6-4335-a899-7d51a9829346
# ╠═11098780-5235-40a1-9477-091ce68420a9
# ╠═f8ff4eca-1d38-45dc-9d85-5e0de774aa4a
# ╠═91636720-2e59-47ab-87ff-a50e9d817d18
# ╠═2cd21c3b-c826-40d6-9bdf-5e808c23a7d3
# ╠═c71c562b-bb4f-4688-a451-daf2040ede62
# ╠═efe707a1-41c7-439d-9e57-dfea374f355d
# ╠═015bdde5-77c9-49d7-a8ba-c9c11f0033a9
# ╠═62a4c01b-434b-4a2b-a46b-78bee5136dad
# ╠═07314336-5096-4fd3-a94a-70e41a7d3a6a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
