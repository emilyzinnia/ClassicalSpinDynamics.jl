using DifferentialEquations
using ClassicalSpinMC: Lattice, read_spin_configuration!, get_bilinear_sites, get_bilinear_matrices, get_onsite, get_field

"""
Time evolve homogeneous ODE. 
"""
function timeEvolve!(du::Vector{Float64}, u::Vector{Float64}, p::Tuple{Lattice, Float64}, t)
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
Time evolve inhomogeneous ODE for spectroscopy.
"""
function timeEvolveTD!(du::Vector{Float64}, u::Vector{Float64}, p::Tuple{Lattice, Float64, Function}, t)
    lat = p[1]
    N = lat.size
    alpha = p[2] 
    B = p[3](t)   
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
        Hx += -B[1]
        Hy += -B[2]
        Hz += -B[3]

        # components are stored in multiples of i
        px= (u[3i-1] * Hz - u[3i] * Hy)
        py= (u[3i] * Hx - u[3i-2] * Hz)
        pz= (u[3i-2] * Hy - u[3i-1] * Hx)
        du[3i-2] = damp_const * (u[3i-1] * (Hz + alpha*pz) - u[3i] * (Hy + alpha*py) )
        du[3i-1] = damp_const * (u[3i] * (Hx + alpha*px) - u[3i-2] * (Hz + alpha*pz))
        du[3i] = damp_const * (u[3i-2] * (Hy + alpha*py) - u[3i-1] * (Hx + alpha*px))
    end
    nothing
end


"""
    compute_St(ts::Array{Float64}, lat::Lattice; 
                    alg=Tsit5(), tol::Float64=1e-7, alpha::Float64=0.0)

Time evolves spin configuration according to LLG equations. 

Returns S(t) matrix with dims=(N_t, 3N) where N_t is the number of timesteps and N is the number of lattice sites.

# Arguments
- `ts::Array{Float64}`: time array 
- `lat::Lattice`: Lattice object containing spin configuration 

# Keyword Arguments
- `alg=Tsit5()`: ODE solver algorithm for DifferentialEquations solver 
- `tol::Float64=1e-7`: tolerance for ODE solver 
- `alpha::Float64=0.0`: damping parameter 
"""
function compute_St(ts::Array{Float64}, lat::Lattice; 
                    alg=Tsit5(), tol::Float64=1e-7, alpha::Float64=0.0, dt::Float64=0.25, kwargs...)

    # time evolve the spins 
    s0 = vcat(lat.spins...)   # flatten to vector of (Sx1, Sy1, Sz1...)
    params = (lat, alpha)
    prob = ODEProblem(timeEvolve!, s0, (min(ts...), max(ts...)), params)
    
    # solve ODE 
    sol = solve(prob, alg, reltol=tol, abstol=tol, save_on=true, dt=dt, kwargs...)

    return hcat(sol.u...)
end

function compute_St(ts::Array{Float64}, pulse::Function, lat::Lattice; 
    alg=Tsit5(), tol::Float64=1e-7, alpha::Float64=0.0, dt::Float64=0.25, kwargs...)

    # time evolve the spins 
    s0 = vcat(lat.spins...)   # flatten to vector of (Sx1, Sy1, Sz1...)
    
    params = (lat, alpha, pulse)
    prob = ODEProblem(timeEvolveTD!, s0, (min(ts...), max(ts...)), params)

    # solve ODE 
    sol = solve(prob, alg, reltol=tol, abstol=tol, save_on=true, dt=dt, kwargs...)

    return hcat(sol.u...)
end
