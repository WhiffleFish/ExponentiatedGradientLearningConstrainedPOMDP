Base.@kwdef struct ExponentiatedGradientSolver{EVAL, O<:NamedTuple} <: Solver
    max_time::Float64   = 1e5
    max_iter::Int       = 100
    max_steps::Int      = typemax(Int)
    evaluator::EVAL     = PolicyGraphEvaluator(max_steps) #MCEvaluator()
    verbose::Bool       = false
    pomdp_sol_options::O= (;delta=0.75)
    B                   = 10
    η                   = 0.1
end


mutable struct ExponentiatedGradientSolverSolution <: Policy
    const C::Matrix{Float64}
    const V::Vector{Float64}
    const dual_vectors::Vector{Float64}
    policy_idx::Int
    const problem::CGCPProblem
    const evaluator::Union{PolicyGraphEvaluator,RecursiveEvaluator,MCEvaluator}
end

function compute_policy(sol, m::CGCPProblem, λ::Vector{Float64})
    m.λ = λ
    s_pol = solve(sol,m)
    return  s_pol
end

function compute_policy(sol::HSVI4CGCP.SARSOPSolver, m::CGCPProblem, λ::Vector{Float64})
    m.λ = λ
    soln = solve_info(sol,m)
    s_pol = soln[1]
    ub = soln[2][:tree].V_upper[1]
    return  s_pol, ub
end

function POMDPs.solve(solver::ExponentiatedGradientSolver, pomdp::CPOMDP)
    t0 = time()
    (;max_time, max_iter, evaluator, verbose, τ, pomdp_sol_options) = solver
    nc = constraint_size(pomdp)
    prob = CGCPProblem(pomdp, ones(nc), false)
    pomdp_solver = HSVI4CGCP.SARSOPSolver(;max_time=τ, max_steps=solver.max_steps, pomdp_sol_options...)
    
    λ = solver.B/2
    iter = 0

    ĉ = first(constraints(m.m))

    C = Matrix{Float64}
    V = Float64[]
    λ_hist = Float64[]

    while time() - t0 < max_time && iter < max_iter
        iter += 1
        pomdp_solver = HSVI4CGCP.SARSOPSolver(;max_time=τ,max_steps=solver.max_steps, pomdp_sol_options...)
        πt,v_ub = compute_policy(pomdp_solver,prob,λ)
        v_t, c_t = evaluate_policy(evaluator, prob, πt)
        
        verbose && println("""
            iteration $iter
            c = $c_t
            v = $v_t
            λ = $λ
            τ = $τ
            δ = $δ
            ϕa = $ϕa
            Δϕ = $(ϕu-ϕl)
        ----------------------------------------------------
        """)

        C = hcat(C, c_t)
        V = push!(V, v_t)
        λ_hist = push!(λ_hist, λ)

        #update lambda
        λ = B*(λ*exp(-η*(ĉ - c_t)))/(B + λ*(exp(-η*(ĉ - c_t)) - 1)) #check if signs are correct
    end
    return ExponentiatedGradientSolverSolution(C, V, λ_hist, 0, prob, evaluator)
end

reset!(p::ExponentiatedGradientSolverSolution) = p.policy_idx = 0

function initialize!(p::ExponentiatedGradientSolverSolution)
    probs = p.p_pi
    p.policy_idx = rand(SparseCat(eachindex(probs), probs))
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
    v = 0.0
    for (π_i, p_i) ∈ zip(p.policy_vector, p.p_pi)
        v += p_i*evaluate_policy(p.evaluator, p.problem, π_i, b)[1]
    end
    return v
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
