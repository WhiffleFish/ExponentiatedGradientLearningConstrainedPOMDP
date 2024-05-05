mutable struct BellmanCounterExPOMDP <: POMDP{Int, Int, Bool}
    p_sense_correctly::Float64
    gold_reward::Float64
    discount::Float64
end

BellmanCounterExPOMDP() = BellmanCounterExPOMDP(1.0, 100.0, 0.99999999999999)

POMDPs.discount(m::BellmanCounterExPOMDP)          = m.discount
POMDPs.states(::BellmanCounterExPOMDP)             = (1:5)
POMDPs.actions(::BellmanCounterExPOMDP)            = 1:2
POMDPs.isterminal(::BellmanCounterExPOMDP, s)      = s === 5

POMDPs.stateindex(::BellmanCounterExPOMDP, s)      = s
POMDPs.actionindex(::BellmanCounterExPOMDP, a)     = a
POMDPs.observations(::BellmanCounterExPOMDP) = (false, true)
POMDPs.obsindex(::BellmanCounterExPOMDP, o::Bool) = Int64(o) + 1

function POMDPs.transition(pomdp::BellmanCounterExPOMDP, s::Int, a::Int)
    if isterminal(pomdp, s)
        return Deterministic(s)
    end

    if s == 1 || s == 2
        if a == 1
            return Deterministic(s+2)
        else
            Deterministic(5)
        end
    elseif s == 3 || s == 4
        return Deterministic(5)
    end
    return Deterministic(5)
end

function POMDPs.observation(pomdp::BellmanCounterExPOMDP, a, sp)
    pc = pomdp.p_sense_correctly
    p = 1.0
    if a == 1
        sp == 3 ? (p = pc) : (p = 1.0-pc)
    else
        p = 0.5
    end
    return BoolDistribution(p)
end

function POMDPs.reward(pomdp::BellmanCounterExPOMDP, s::Int, a::Int)

    if isterminal(pomdp,s)
        return 0.0
    end
    r = 0.0
    if s == 1 || s == 2
        if a == 2
            r += 10.0
        end
    end
    if s == 3
        if a == 1
            r += 12.0
        elseif a == 2
            r += 0
        end
    elseif s == 4
        if a == 1
            r = 12
        elseif a == 2
            r = 0
        end
    end
    return r
end

# POMDPs.initialstate(pomdp::BellmanCounterExPOMDP) = Uniform([1,2])
POMDPs.initialstate(pomdp::BellmanCounterExPOMDP) = SparseCat([1,2], [0.2, 0.8])
##Constrained
struct BellmanCounterExCPOMDP{V<:AbstractVector} <: CPOMDP{Int,Int,Bool}
    m::BellmanCounterExPOMDP
    constraints::V
end

@POMDP_forward BellmanCounterExCPOMDP.m

ĉ = [0.0]

BellmanCounterExCPOMDP(ĉ=[1.0]; kwargs...) = BellmanCounterExCPOMDP(BellmanCounterExPOMDP(;kwargs...),ĉ)

ConstrainedPOMDPs.constraints(p::BellmanCounterExCPOMDP) = p.constraints

function ConstrainedPOMDPs.costs(constrained::BellmanCounterExCPOMDP, s, a)
    c = 0.0
    if s == 1 || s == 2
        if a == 2
            c = 5.0
        end
    end
    if s == 3
        if a == 1
            c = 9.0
        elseif a == 2
            c = 5.0
        end
    end
    if s == 4
        if a == 1
            c = 4.0
        elseif a == 2
            c = 5.0
        end
    end

    return c
end