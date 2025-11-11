using DifferentialEquations
using Dates
using BinningAnalysis
using HDF5
using ProgressMeter

struct MDbuffer
    tstep::Float64
    tmin::Float64
    tmax::Float64
    tt::Array{Float64, 1}
    freq::Array{Float64, 1}
    ks::Array{Float64, 2}
    alpha::Float64
end

function get_tt_freq(tstep::Float64=0.01, tmin::Float64=0.0, tmax::Float64=0.5)
    tt = collect(tmin:tstep:tmax)
    N_time = length(tt)
    N = iseven(N_time) ? N_time : N_time+1
    freq = [ n/(N*tstep) for n in 0:N ] 
    return tt, freq
end 

function MD_buffer(ks::Array{Float64}, tstep::Float64=0.01, tmin::Float64=0.0, tmax::Float64=100.0, alpha::Float64=0.0)::MDbuffer
    # frequencies
    tt, freq = get_tt_freq(tstep, tmin, tmax)
    return MDbuffer(tstep, tmin, tmax, tt, freq, ks, alpha)
end

"""
Computes S(q,ω) from spin configuration
"""
function compute_Sqw(lat::Lattice, md::MDbuffer, alg=Tsit5(), tol::Float64=1e-7)
    # time evolve the spins 
    s0 = vcat(lat.spins...)   # flatten to vector of (Sx1, Sy1, Sz1...)
    params = (lat, md.alpha)
    ks = md.ks
    N_k = size(ks)[2]
    pos = lat.site_positions
    omega = md.freq 
    Sqw = zeros(ComplexF64, 3N_k, length(omega))
    spins = zeros(Float64, 3, lat.size)

    if (length(lat.cubic_sites[1]) != 0) | (length(lat.quartic_sites[1]) != 0) 
        timeEvolution = timeEvolveHigherOrder!
    else
        timeEvolution = timeEvolve!
    end

    function perform_measurements!(integrator)
        t = integrator.t
        spins[1,:] .= integrator.u[3 * collect(1:lat.size) .- 2] #sx
        spins[2,:] .= integrator.u[3 * collect(1:lat.size) .- 1] #sy
        spins[3,:] .= integrator.u[3 * collect(1:lat.size) ]     #sz

        for n=1:N_k
            phase = exp.(-im * transpose(ks[:, n]) * pos)
            sqx = (phase * spins[1,:])[1] 
            sqy = (phase * spins[2,:])[1] 
            sqz = (phase * spins[3,:])[1] 
            for w in 1:length(omega)
                Sqw[3n-2, w] += sqx * exp(im * omega[w] * t)
                Sqw[3n-1, w] += sqy * exp(im * omega[w] * t)
                Sqw[3n  , w] += sqz * exp(im * omega[w] * t)
            end
        end
    end

    prob = ODEProblem(timeEvolution, s0, (md.tmin, md.tmax), params)
    cb = PresetTimeCallback(md.tt, perform_measurements!)
    
    # solve ODE 
    sol = solve(prob, alg, reltol=tol, abstol=tol, callback=cb, dense=false, save_on=false, dt=md.tstep)

    # normalize S(q,ω)
    norm = sqrt(length(md.tt) * 2 * pi * lat.size )
    Sqw ./= norm

    return Sqw[3 * collect(1:N_k) .- 2, :], Sqw[3 * collect(1:N_k) .- 1, :], Sqw[3 * collect(1:N_k) , :] 
end

# do this step in parallel 
function compute_FT_correlations(S_q::NTuple{3, Array{ComplexF64, 2}})
    sx, sy, sz = S_q 
    Suv = zeros(Float64, 9, size(sx)...) # store SSF results
    
    sx_ = conj.(sx)
    sy_ = conj.(sy)
    sz_ = conj.(sz)

    # compute correlations
    Suv[1, :, :] .= real.(sx .* sx_)
    Suv[2, :, :] .= real.(sx .* sy_)
    Suv[3, :, :] .= real.(sx .* sz_)
    Suv[4, :, :] .= real.(sy .* sx_)
    Suv[5, :, :] .= real.(sy .* sy_)
    Suv[6, :, :] .= real.(sy .* sz_)
    Suv[7, :, :] .= real.(sz .* sx_)
    Suv[8, :, :] .= real.(sz .* sy_)
    Suv[9, :, :] .= real.(sz .* sz_)
    return  Suv 
end

"""
    runDSSF!(path::String, tstep::Real, tmin::Real, tmax::Real, lat::Lattice, 
                               ks::Matrix{Float64}; alg=Tsit5(), tol::Float64=1e-7,
                               override=false; alpha::Float64=0.0)

Time evolves each spin configuration in provided path using LLG equations.

Computes and writes out dynamical spin correlation for each configuration. 

# Arguments:
- `path::String`: path to folder containing initial configurations with trailing backslash 
- `tstep::Real`: timestep 
- `tmin::Real`: minimum of time interval
- `tmax::Real`: max of time interval
- `lat::Lattice`: Lattice object 
- `ks::Matrix{Float64}`: matrix containing wavevectors 

# Keyword arguments:
- `alg=Tsit5()`: algorithm used for the `DifferentialEquations.jl` ODE solver 
- `tol::Float64=1e-7`: tolerance for solver 
- `override=false`: flag to overwrite configuration with existing MD results 
- `alpha::Float64=0.0`: damping parameter 
"""
function runDSSF!(path::String, tstep::Real, tmin::Real, tmax::Real, lat::Lattice, 
                    ks::Matrix{Float64}; alg=Tsit5(), tol::Float64=1e-7,
                    override=false, alpha::Float64=0.0)
    # initialize MPI parameters 
    rank = 0
    commSize = 1
    enableMPI = false
    
    if MPI.Initialized()
        comm = MPI.COMM_WORLD
        commSize = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        if commSize > 1
            enableMPI = true
        end
    end

    # split computations evenly along all nodes 
    N_IC = length(readdir(path)) # number of initial configurations 
    N_rank = Array{Int64, 1}(undef, commSize)
    N_per_rank = N_IC ÷ commSize 
    N_remaining = N_IC % commSize

    # distribute remaining configurations 
    if (N_remaining != 0) && rank < N_remaining
        N_per_rank += 1
    end 

    N_rank[rank+1] = N_per_rank
    commSize > 1 && MPI.Allgather!(MPI.IN_PLACE, N_rank, 1, MPI.COMM_WORLD) 

    IC = rank == 0 ? 0 : sum(N_rank[1:rank])
    MD = MD_buffer(ks, tstep, tmin, tmax, alpha)

    for i in 1:N_per_rank
        # initialize lattice object from hdf5 file 
        file = string(path, "IC_$IC.h5") 
        read_spin_configuration!(lat, file)
        
        f = h5open(file, "r+") 
        exists = haskey(f, "spin_correlations")
        close(f)
        if exists && !override
            println("Skipping IC_$IC")
            IC += 1
            continue
        else
            println("Time evolving for IC $i/$N_per_rank on rank $rank")
            @time S_qw = compute_Sqw(lat, MD, alg, tol)

            # compute total correlations 
            corr = compute_FT_correlations(S_qw)
            println("Writing IC $IC to file on rank $rank")
            res = Dict("corr"=>corr, "freq"=>MD.freq, "momentum"=>ks, "S_qw"=>S_qw)
            h5open(file, "r+") do f
                g = haskey(f, "spin_correlations") ? f["spin_correlations"] : create_group(f, "spin_correlations")
                overwrite_keys!(g, res)
            end
        end
        # increment configuration 
        IC += 1
    end

    println("Calculation completed on rank $rank on ", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
end

"""
Computes dynamical spin structure factor. 

Averages over all spin correlations in a given path. 

# Arguments:
- `path::String`: path to initial configuration files 
- `dest::String`: path to write out final DSSF to 
- `params::Dict{String,Float64}`: human readable dictionary of parameters to write out 
"""
function compute_dynamical_structure_factor(path::String, dest::String, params::Dict{String, <:Any}=Dict{String,<:Any}())
    # initialize LogBinner
    println("Initializing LogBinner in $path")
    files = readdir(path)
    f0 = h5open(string(path, files[1]), "r")
    shape = size(read(f0["spin_correlations/corr"]) )
    shape_Sq = size(read(f0["spin_correlations/S_qw"]) )
    ks = read(f0["spin_correlations/momentum"])
    freq = read(f0["spin_correlations/freq"])
    close(f0)
    DSF = LogBinner(zeros(Float64, shape...))
    DSF_disc = LogBinner(zeros(Float64, shape_Sq...))

    # collecting correlations
    println("Collecting correlations from ", length(files), " files")
    for file in files 
        h5open(string(path, file), "r") do f 
            Suv = read(f["spin_correlations/corr"])
            Sqw = read(f["spin_correlations/S_qw"])
            push!(DSF, Suv)
            push!(DSF_disc, Sqw)
        end
    end

    total = mean(DSF)
    disc = compute_FT(mean(DSF_disc))
    conn = total .- disc
    # write to configuration file 
    println("Writing to $dest")
    d = h5open(dest, "r+")
    res = Dict( "freq"=>freq, "momentum"=>ks,  "total"=>total, 
                "disconnected"=> disc, "connected"=> conn)
    g = haskey(d, "spin_correlations") ? d["spin_correlations"] : create_group(d, "spin_correlations")
    overwrite_keys!(g, params)
    overwrite_keys!(g, res)
    close(d)

    println("Successfully averaged ",length(files), " files")
end
