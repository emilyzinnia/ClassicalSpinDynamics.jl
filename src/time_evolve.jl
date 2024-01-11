using DifferentialEquations
using ClassicalSpinMC: Lattice, read_spin_configuration!, get_bilinear_sites, get_bilinear_matrices, get_onsite, get_field

"""
Time evolve homogeneous ODE. 
"""
function timeEvolve!(du, u, p, t)
    lat = p[1]
    N = lat.size
    alpha = p[2] 
    damp_const = 1/(1+alpha^2)
    for i=1:N
        js = get_bilinear_sites(lat,i)
        Js = get_bilinear_matrices(lat,i)
        h = get_field(lat,i)
        o = get_onsite(lat,i)

        Hx = 0.0
        Hy = 0.0
        Hz = 0.0

        # on-site interaction
        six = u[3i-2]
        siy = u[3i-1]
        siz = u[3i]
        Hx += 2 * ( o.m11 * six + o.m12 * siy + o.m13 * siz)
        Hy += 2 * ( o.m21 * six + o.m22 * siy + o.m23 * siz)
        Hz += 2 * ( o.m31 * six + o.m32 * siy + o.m33 * siz)

        # bilinear local effective field 
        for n in eachindex(js)
            J = Js[n]
            j = js[n]
            ux = u[3j-2]
            uy = u[3j-1]
            uz = u[3j]

            Hx += (J.m11 * ux + J.m12 * uy + J.m13 * uz)
            Hy += (J.m21 * ux + J.m22 * uy + J.m23 * uz)
            Hz += (J.m31 * ux + J.m32 * uy + J.m33 * uz)
        end
        # Zeeman from static field 
        Hx += -h[1]
        Hy += -h[2]
        Hz += -h[3] 

        # components are stored in multiples of i
        px= (u[3i-1] * Hz - u[3i] * Hy)
        py= (u[3i] * Hx - u[3i-2] * Hz)
        pz= (u[3i-2] * Hy - u[3i-1] * Hx)
        du[3i-2] = damp_const * (u[3i-1] * (Hz + alpha*pz) - u[3i] * (Hy + alpha*py) )
        du[3i-1] = damp_const * (u[3i] * (Hx + alpha*px) - u[3i-2] * (Hz + alpha*pz))
        du[3i] = damp_const * (u[3i-2] * (Hy + alpha*py) - u[3i-1] * (Hx + alpha*px))
    end
end

"""
Time evolve inhomogeneous ODE for 2D spectroscopy.
"""
function timeEvolve2D!(du::Vector{Float64}, u::Vector{Float64}, p, t)
    lat = p[1]
    N = lat.size
    alpha = p[2] 
    tau = p[3]
    BA = p[4](t+tau)        # first pulse
    BB = p[5](t)    # second pulse 
    damp_const = 1/(1+alpha^2)
    for i=1:N
        js = get_bilinear_sites(lat,i)
        Js = get_bilinear_matrices(lat,i)
        h = get_field(lat,i)
        o = get_onsite(lat,i)

        Hx = 0.0
        Hy = 0.0
        Hz = 0.0

        # on-site interaction
        six = u[3i-2]
        siy = u[3i-1]
        siz = u[3i]
        Hx += 2 * ( o.m11 * six + o.m12 * siy + o.m13 * siz)
        Hy += 2 * ( o.m21 * six + o.m22 * siy + o.m23 * siz)
        Hz += 2 * ( o.m31 * six + o.m32 * siy + o.m33 * siz)

        # bilinear local effective field 
        for n in eachindex(js)
            J = Js[n]
            j = js[n]
            ux = u[3j-2]
            uy = u[3j-1]
            uz = u[3j]

            Hx += (J.m11 * ux + J.m12 * uy + J.m13 * uz)
            Hy += (J.m21 * ux + J.m22 * uy + J.m23 * uz)
            Hz += (J.m31 * ux + J.m32 * uy + J.m33 * uz)
        end
        # Zeeman from static field 
        Hx += -h[1]
        Hy += -h[2]
        Hz += -h[3] 

        # Zeeman from time-dependent pump/probe field 
        Hx += -BA[1]-BB[1]
        Hy += -BA[2]-BB[2]
        Hz += -BA[3]-BB[3]

        # components are stored in multiples of i
        px= (u[3i-1] * Hz - u[3i] * Hy)
        py= (u[3i] * Hx - u[3i-2] * Hz)
        pz= (u[3i-2] * Hy - u[3i-1] * Hx)
        du[3i-2] = damp_const * (u[3i-1] * (Hz + alpha*pz) - u[3i] * (Hy + alpha*py) )
        du[3i-1] = damp_const * (u[3i] * (Hx + alpha*px) - u[3i-2] * (Hz + alpha*pz))
        du[3i] = damp_const * (u[3i-2] * (Hy + alpha*py) - u[3i-1] * (Hx + alpha*px))
    end
end

"""
    compute_St(ts::Array{Float64}, lat::Lattice; 
                    alg=Tsit5(), tol::Float64=1e-7, alpha::Float64=0.0, specparams::Dict{String,<:Any}=Dict{String,<:Any}())

Time evolves spin configuration according to LLG equations. 

Returns S(t) matrix with dims=(N_t, 3N) where N_t is the number of timesteps and N is the number of lattice sites.

Will use `timeEvolve!` by default; to use `timeEvolve2D!`, specify `specparams`. 

# Arguments
- `ts::Array{Float64}`: time array 
- `lat::Lattice`: Lattice object containing spin configuration 

# Keyword Arguments
- `alg=Tsit5()`: ODE solver algorithm for DifferentialEquations solver 
- `tol::Float64=1e-7`: tolerance for ODE solver 
- `alpha::Float64=0.0`: damping parameter 
- `specparams::Dict{String,Any}=Dict{String,Any}()`: if doing 2D spectroscopy, feed dictionary with the keys
    -`tau::Float64`
    -`pulseA::Function`
    -`pulseB::Function`
"""
function compute_St(ts::Array{Float64}, lat::Lattice; 
                    alg=Tsit5(), tol::Float64=1e-7, alpha::Float64=0.0, specparams::Dict{String,<:Any}=Dict{String,Any}())

    # time evolve the spins 
    s0 = vcat(lat.spins...)   # flatten to vector of (Sx1, Sy1, Sz1...)
    St = zeros(Float64, length(ts), 3*lat.size)
    params = [lat, alpha]

    if length(specparams) != 0
        push!(params, specparams["tau"])
        push!(params, specparams["pulseA"])
        push!(params, specparams["pulseB"])
        timeevolve = timeEvolve2D!
    else 
        timeevolve = timeEvolve!
    end

    count = 1
    function perform_measurements!(integrator)
        St[count,:] .= integrator.u
        count += 1
    end

    prob = ODEProblem(timeevolve, s0, (min(ts...), max(ts...)), params)
    cb = PresetTimeCallback(ts, perform_measurements!)
    
    # solve ODE 
    solve(prob, alg, reltol=tol, abstol=tol, callback=cb, dense=false, save_on=false)

    return St 
end
