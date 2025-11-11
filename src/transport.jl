using DifferentialEquations
using Dates
using BinningAnalysis
using HDF5
using ProgressMeter
using LinearAlgebra
using ClassicalSpinMC: Lattice, get_interaction_sites, get_interaction_matrices, get_field, timeEvolve!, read_spin_configuration!
using Logging, LoggingExtras
using Einsum
include("stack.jl")

function crosst(a::NTuple{3,Real}, b::NTuple{3,Real})
    return ( a[2]*b[3]-a[3]*b[2], a[3]*b[1]-a[1]*b[3], a[1]*b[2]-a[2]*b[1] )
end

function interaction_matrix(Hij)
    return [Hij.m11 Hij.m12 Hij.m13
    Hij.m21 Hij.m22 Hij.m23
    Hij.m31 Hij.m32 Hij.m33]
end

"""
Computes the energy current for cubic interactions. Note: Presence of both cubic and Zeeman interactions is not supported.
"""
function compute_energy_current(i::Int64, lat::Lattice, St::Array{Float64,1})::Tuple{Float64, Float64}
    js = lat.interaction_sites[i]
    Js = lat.interaction_matrices[i]
    hi = get_field(lat, i)
    Si = (St[3i-2], St[3i-1], St[3i])
    pos = lat.site_positions
    rij = zeros(Float64, 2)
    ji = zeros(Float64, 2)
    for n in eachindex(js)
        Hij = Js[n]
        j = js[n]
        Sj = (St[3j-2], St[3j-1], St[3j])
        rij .= pos[:,j] - pos[:,i]
        Jij = 0.0

        # (HijSj)
        pre0 = (Hij.m11*Sj[1]+Hij.m12*Sj[2]+Hij.m13*Sj[3], 
                Hij.m21*Sj[1]+Hij.m22*Sj[2]+Hij.m23*Sj[3], 
                Hij.m31*Sj[1]+Hij.m32*Sj[2]+Hij.m33*Sj[3])

        # Si x (HijSj)
        pre1 = crosst(Si, pre0)

        # transpose(Si)Hij
        pre2 = (Hij.m11*Si[1]+Hij.m21*Si[2]+Hij.m31*Si[3], 
                Hij.m12*Si[1]+Hij.m22*Si[2]+Hij.m32*Si[3], 
                Hij.m13*Si[1]+Hij.m23*Si[2]+Hij.m33*Si[3])
        
        ks = lat.interaction_sites[j]
        Jjks = lat.interaction_matrices[j]
        
        for m in eachindex(js)
            # first term, sum over l belonging to i 
            Hil = Js[m]
            l = js[m]
            Sl = (St[3l-2], St[3l-1], St[3l])
            Jij += 2*dot(pre1, (Hil.m11*Sl[1]+Hil.m12*Sl[2]+Hil.m13*Sl[3], 
                              Hil.m21*Sl[1]+Hil.m22*Sl[2]+Hil.m23*Sl[3], 
                              Hil.m31*Sl[1]+Hil.m32*Sl[2]+Hil.m33*Sl[3]))

            # second term, sum over k belonging to j 
            Hjk = Jjks[m]
            k = ks[m]
            Sk = (St[3k-2], St[3k-1], St[3k])
            Jij += 2*dot(pre2, crosst(Sj, (Hjk.m11*Sk[1]+Hjk.m12*Sk[2]+Hjk.m13*Sk[3], 
                                         Hjk.m21*Sk[1]+Hjk.m22*Sk[2]+Hjk.m23*Sk[3], 
                                         Hjk.m31*Sk[1]+Hjk.m32*Sk[2]+Hjk.m33*Sk[3])))
        
        end
        # field term 
        hj = get_field(lat, j)
        Jij += dot(pre0, crosst(Si, hi)) - dot(pre2, crosst(Sj, hj)) 
        ji .+= rij * Jij
    end

    return ji[1], ji[2]
end

function compute_total_energy_current(lat::Lattice, St::Array{Float64,1})::Tuple{Float64, Float64}
    # divide by two since we are double counting the current over the sites 
    jEx = 0.0
    jEy = 0.0
    for i=1:lat.size
        jx, jy = compute_energy_current(i, lat, St)
        jEx += jx
        jEy += jy
    end
    return jEx/2, jEy/2
end

function compute_current_correlation(lat::Lattice, St::Matrix{Float64})
    S0 = St[1,:]
    j0 = compute_total_energy_current(lat, S0)
    jxy = zeros(Float64, size(St)[1])
    jyx = zeros(Float64, size(St)[1])
    for t in 1:size(St)[1]
        jt = compute_total_energy_current(lat, St[t,:])
        jxy[t] = jt[2] * j0[1]
        jyx[t] = jt[1] * j0[2]

    end
    return jxy, jyx
end

# the full term is calculated in thermal_hall.jl where averaged and divided by T
function compute_energy_magnetization(lat::Lattice)
    N = lat.size 
    pos = lat.site_positions
    spins = vcat(lat.spins...)
    jExy = 0.0
    jEyx = 0.0

    for i=1:N
        Ji = compute_energy_current(i, lat, spins)
        jExy += pos[2, i] * Ji[1]
        jEyx += pos[1, i] * Ji[2]
    end
    return -2 * jExy , -2 * jEyx
end


function compute_energy_current_cubic(i::Int64, lat::Lattice, St::Array{Float64,1})::Tuple{Float64, Float64}
    js = lat.interaction_sites[i]
    Js = lat.interaction_matrices[i]
    cs = lat.cubic_sites[i]
    Cs = lat.cubic_matrices[i]

    Si = (St[3i-2], St[3i-1], St[3i])
    pos = lat.site_positions
    rij = zeros(Float64, 2)
    ji = zeros(Float64, 2)

    Hi = get_local_field(lat, i)
    
    # quadratic term 
    for n in eachindex(js)
        Hij = Js[n]
        j = js[n]
        Sj = (St[3j-2], St[3j-1], St[3j])
        rij .= pos[:,j] - pos[:,i]
        Jij = 0.0
        Hj = get_local_field(lat, j)
        
        # (HijSj)
        pre0 = (Hij.m11*Sj[1]+Hij.m12*Sj[2]+Hij.m13*Sj[3], 
                Hij.m21*Sj[1]+Hij.m22*Sj[2]+Hij.m23*Sj[3], 
                Hij.m31*Sj[1]+Hij.m32*Sj[2]+Hij.m33*Sj[3])

        # transpose(Si)Hij
        pre1 = (Hij.m11*Si[1]+Hij.m21*Si[2]+Hij.m31*Si[3], 
                Hij.m12*Si[1]+Hij.m22*Si[2]+Hij.m32*Si[3], 
                Hij.m13*Si[1]+Hij.m23*Si[2]+Hij.m33*Si[3])

        Jij += 1/2 * (dot(Hi, crosst(pre0, Si)) +  dot(Hj, crosst(pre1, Sj)))

        ji .+= rij * Jij
    end

    # cubic term
    for n in eachindex(cs)
        C = Cs[n]
        j, k = cs[n]
        if cs[n] == (0,0)
            continue
        end
        Sj = (St[3j-2], St[3j-1], St[3j])
        Sk = (St[3k-2], St[3k-1], St[3k])
        rij .= pos[:,j] - pos[:,i]
        Jij = 0.0

        Hj = get_local_field(lat, j)
        Hk = get_local_field(lat, k)

        @einsum Ci := C[a, b, c] * Sj[b] * Sk[c] 
        @einsum Cj := C[a, b, c] * Si[a] * Sk[c]
        @einsum Ck := C[a, b, c] * Si[a] * Sj[b]
        Jij += 1/3 *  (dot( crosst(Si, Hi), Ci) + 
                       dot( crosst(Sj, Hj), Cj) + 
                       dot( crosst(Sk, Hk), Ck) )

        ji .+= rij * Jij
    end

    return ji[1], ji[2]
end

function runTransportSingle!(file, dt, tmax, lat::Lattice, alpha::Float64=0.01, override=false, kubo=true)
    read_spin_configuration!(file, lat)
    f = h5open(file, "r+") 
    exists = haskey(f, "transport")
    close(f)
    if exists && !override
        println("Skipping $file")
        return 
    end 

    # compute site dependent energy currents 
    ji = zeros(Float64,2,lat.size)
    for i in 1:lat.size
        jx, jy = compute_energy_current(i,lat,vcat(lat.spins...))
        ji[1,i] = jx
        ji[2,i] = jy
    end
    res = Dict{String, Any}("ji"=>ji)

    if kubo
        # compute time dependence 
        println("Computing currents for $file")
        @time tt, St = compute_St(lat, dt, tmax, alpha)

        # compute current correlations 
        println("Computing current correlations for $file")
        @time jxjy, jyjx = compute_current_correlation(lat, St)

        # compute energy magnetization currents
        println("Computing energy magnetization current for $file")
        @time jExy, jEyx = compute_energy_magnetization(lat)

        merge!(res, Dict( "jxjy"=>jxjy,"jyjx"=>jyjx,
                            "jExy"=>jExy, "jEyx"=>jEyx, "t"=>tt) )
    end

    println("Writing to $file")
    h5open(file, "r+") do f
        if haskey(f, "transport")
            g = f["transport"]
            overwrite_keys!(g, res)
        else
            g = create_group(f, "transport")
            for key in keys(res)
                g[key] = res[key]
            end
        end
    end
end

function runTransportBatch!(path, dt, tmax, lat::Lattice; alpha::Float64=0.01, override=false, kubo=true)
    #initialize MPI
    MPI.Initialized() || throw(ErrorException("MPI has not been initialized"))
    comm = MPI.COMM_WORLD
    commSize = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

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

    for i in 1:N_per_rank
        # initialize lattice object from hdf5 file 
        file = string(path, "IC_$IC.h5") 
        println("Running transport for IC $IC/$N_per_rank")
        runTransportSingle!(file, dt, tmax, lat, alpha, override, kubo)
        # increment configuration 
        IC += 1
    end
    println("Calculation completed on rank $rank on ", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
end


function runTransportStack!(stackfile, dt, tmax, lat::Lattice; alpha::Float64=0.01, override=false, kubo=true)
    # check if MPI initialized 
    if MPI.Initialized()
        comm = MPI.COMM_WORLD
        commSize = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
    else
        commSize = 1 
        rank = 0 
    end

    # read spin configurtion based on file from the stack 
    file = pullFromStack!(stackfile)
    logger = FileLogger("$stackfile.log"; append=true)
    with_logger(logger) do 
        while length(file) != 0
            try
                @info runTransportSingle!(file, dt, tmax, lat, alpha, override, kubo)
            catch 
                println("ERROR: ", error("Failed on $file, pushing back to stack"))
                pushToStack!(stackfile, file)
                stacktrace(catch_backtrace())
            end
            
            file = pullFromStack!(stackfile) # pull next file from stack 
            attempts = 0 
            while length(file) == 0
                println("Stack empty; idling")
                sleep(20)
                file = pullFromStack!(stackfile)
                attempts += 1
                if attempts > 10
                    println("Stack empty; relinquishing job")
                    return 
                end 
            end
        end
    end
end 
