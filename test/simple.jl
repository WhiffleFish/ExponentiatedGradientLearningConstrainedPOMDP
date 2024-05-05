begin
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    using ExponentiatedGradient
    Pkg.activate(@__DIR__)
    using POMDPs
    using POMDPTools
    using POMDPModels
    using ProgressMeter
    include("Envs/Envs.jl")
    using .Envs
end

cpomdp = BellmanCounterExCPOMDP([5.0])

sol = ExponentiatedGradientSolver()
pol = solve(sol, cpomdp)

N = 100
sim_rewards = zeros(N)
sim_costs = zeros(N)

@showprogress for i ∈ 1:N
    p_idx = rand(SparseCat(eachindex(pol.policy_vector), pol.p_pi))
    policy = pol.policy_vector[p_idx]
    simy = RolloutSimulator(max_steps = 20)
    r,c = CGCP.simulate(simy, cpomdp, policy, DiscreteUpdater(cpomdp), initialstate(cpomdp))
    sim_rewards[i] = r
    sim_costs[i] = only(c)
end

using Plots
using StatsPlots

@show mean(sim_costs)
@show mean(sim_rewards)

default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

p1 = violin([""], sim_rewards; ylabel="reward", show_mean=true, bandwidth=1.0)
p2 = violin([""], sim_costs; ylabel="cost", c=:red, show_mean=true, bandwidth=0.05)
p = plot(p1,p2, dpi=300)

savefig(p, "CGCP_violin.png")
