Base.@kwdef struct ExponentiatedGradientSolver{EVAL, O<:NamedTuple} <: Solver
    max_time::Float64   = 1e5
    max_iter::Int       = 100
    max_steps::Int      = typemax(Int)
    evaluator::EVAL     = PolicyGraphEvaluator(max_steps) #MCEvaluator()
    verbose::Bool       = false
    pomdp_sol_options::O= (;delta=0.75)
    B::Float64          = 10.0
    η::Float64          = 0.1
end

mutable struct ExponentiatedGradientSolverSolution <: Policy
    const policy_vector::Vector{AlphaVectorPolicy}
    const C::Matrix{Float64}
    const V::Vector{Float64}
    const dual_vectors::Matrix{Float64}
    policy_idx::Int
    const problem::CGCPProblem
    const evaluator::Union{PolicyGraphEvaluator,RecursiveEvaluator,MCEvaluator}
end

function compute_policy(sol, m::CGCPProblem, λ::Vector{Float64})
    m.λ = λ
    s_pol = solve(sol,m)
    return s_pol
end

function compute_policy(sol::HSVI4CGCP.SARSOPSolver, m::CGCPProblem, λ::Vector{Float64})
    m.λ = λ
    pol, info = solve_info(sol,m)
    ub = info[:tree].V_upper[1]
    return pol, ub
end

function POMDPs.solve(solver::ExponentiatedGradientSolver, pomdp::CPOMDP)
    t0 = time()
    (;max_time, max_iter, evaluator, verbose, pomdp_sol_options, B) = solver
    nc = constraint_size(pomdp)
    prob = CGCPProblem(pomdp, ones(nc), true) # initialized=true --- OTHERWISE REWARDS ARE NEVER CONSIDERED
    pomdp_solver = HSVI4CGCP.SARSOPSolver(;max_time=max_time, max_steps=solver.max_steps, pomdp_sol_options...)
    Π = AlphaVectorPolicy[]
    λ = [B/2]
    iter = 0

    ĉ = first(constraints(pomdp))

    C = Matrix{Float64}(undef, 1, 0)
    V = Float64[]
    λ_hist = Matrix{Float64}(undef, 1,0)

    while time() - t0 < max_time && iter < max_iter
        η = sqrt(log(2)/2*iter*B^2)
        iter += 1
        πt,v_ub = compute_policy(pomdp_solver,prob,λ)
        v_t, c_t = evaluate_policy(evaluator, prob, πt)
        
        λk = only(λ)
        _c_t = only(c_t)

        verbose && println("""
            iteration $iter
            c = $_c_t
            v = $v_t
            λ = $λk
        ----------------------------------------------------
        """)

        C = hcat(C, c_t)
        V = push!(V, v_t)
        push!(Π, πt)
        λ_hist = hcat(λ_hist, λ)

        #update lambda
        
        λ = [ B * ( λk*exp(-η*(ĉ - _c_t)) )/
            (B + λk*(exp(-η*(ĉ - _c_t)) - 1))
        ] #check if signs are correct
    end

    return ExponentiatedGradientSolverSolution(Π, C, V, λ_hist, 0, prob, evaluator)
end

reset!(p::ExponentiatedGradientSolverSolution) = p.policy_idx = 0

function initialize!(p::ExponentiatedGradientSolverSolution)
    p.policy_idx = rand(eachindex(p.policy_vector))
    p
end

function POMDPs.action(p::ExponentiatedGradientSolverSolution, b)
    iszero(p.policy_idx) && initialize!(p)
    return action(p.policy_vector[p.policy_idx], b)
end

function POMDPs.value(p::ExponentiatedGradientSolverSolution, b)
    return iszero(p.policy_idx) ? probabilistic_value(p, b) : deterministic_value(p, b)
end

function probabilistic_value(p::ExponentiatedGradientSolverSolution, b)
    return sum(p.policy_vector) do πt
        evaluate_policy(p.evaluator, p.problem, πt, b)[1]
    end / length(p.policy_vector)
end

function deterministic_value(p::ExponentiatedGradientSolverSolution, b)
    @assert !iszero(p.policy_idx)
    return evaluate_policy(p.evaluator, p.problem, p.policy_vector[p.policy_idx], b)[1]
end

function lagrange_probabilistic_value(p::ExponentiatedGradientSolverSolution, b)
    v = 0.0
    for (π_i, p_i) ∈ zip(p.policy_vector, p.p_pi)
        v += p_i*POMDPs.value(π_i, b)
    end
    return v
end

function lagrange_deterministic_value(p::ExponentiatedGradientSolverSolution, b)
    @assert !iszero(p.policy_idx)
    return POMDPs.value(p.policy_vector[p.policy_idx], b)
end
