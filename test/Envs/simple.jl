mutable struct SimplePOMDP <: POMDP{Int, Int, Bool}
    p_sense_correctly::Float64
    gold_reward::Float64
    discount::Float64
end

SimplePOMDP() = SimplePOMDP(1.0, 100.0, 0.95)

POMDPs.discount(m::SimplePOMDP)          = m.discount
POMDPs.states(::SimplePOMDP)             = (1:5)
POMDPs.actions(::SimplePOMDP)            = 1:4
POMDPs.isterminal(::SimplePOMDP, s)      = s === 5

POMDPs.stateindex(::SimplePOMDP, s)      = s
POMDPs.actionindex(::SimplePOMDP, a)     = a
POMDPs.observations(::SimplePOMDP) = (false, true)
POMDPs.obsindex(::SimplePOMDP, o::Bool) = Int64(o) + 1
# initialobs(p::SimplePOMDP, s::Bool) = observation(p, 1, s) # listen 

const SENSE = 1
const DRIVE = 2
const FLY = 3
const END = 4

function POMDPs.transition(pomdp::SimplePOMDP, s::Int, a::Int)
    if s == 5
        return Deterministic(s)
    end

    if s == 3 || s == 4
        return Deterministic(5)
    end
    if a == SENSE
        return Deterministic(s)
    elseif a == END
            return Deterministic(5)
    elseif a == FLY
        return Deterministic(s+2)
    elseif a == DRIVE
        return s == 1 ? Deterministic(1) : Deterministic(s+2)
    end

    return Deterministic(s)
end

function POMDPs.observation(pomdp::SimplePOMDP, a, sp)
    pc = pomdp.p_sense_correctly
    p = 1.0
    if a == SENSE
        sp == 2 ? (p = pc) : (p = 1.0-pc)
    elseif a == DRIVE
        sp == 2 ? (p = (pc - 0.2)) : (p = 1.0-(pc-0.2))
    else
        p = 0.5
    end
    return BoolDistribution(p)
end

function POMDPs.reward(pomdp::SimplePOMDP, s::Int, a::Int)
    r = 0.0
    if (a == DRIVE && s == 1)
        r -= 30
    end
    if (a == FLY)
        r -= 40
    end
    if s == 3 || s == 4
        r += pomdp.gold_reward
    end
    return r
end

reward(pomdp::SimplePOMDP, s::Int, a::Int64, sp::Int) = reward(pomdp, s, a)

POMDPs.initialstate(pomdp::SimplePOMDP) = Uniform([1,2])
# POMDPs.initialstate(pomdp::SimplePOMDP) = SparseCat([1,2], [0.8, 0.2])
##Constrained
struct SimpleCPOMDP{V<:AbstractVector} <: CPOMDP{Int,Int,Bool}
    m::SimplePOMDP
    constraints::V
end

@POMDP_forward SimpleCPOMDP.m

ĉ = [6.0]
SimpleCPOMDP(ĉ=[6.0]; kwargs...) = SimpleCPOMDP(SimplePOMDP(;kwargs...),ĉ)

ConstrainedPOMDPs.constraints(p::SimpleCPOMDP) = p.constraints

function ConstrainedPOMDPs.costs(constrained::SimpleCPOMDP, s, a)
    if a == SENSE
        return 1
    elseif a == DRIVE && s == 2
        return 3
    elseif a == DRIVE && s == 1
        return 3
    elseif a == DRIVE
        return 3
    elseif a == FLY
        return 6
    end

    return 0
end