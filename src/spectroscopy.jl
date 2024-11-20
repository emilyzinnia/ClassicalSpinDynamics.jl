using HDF5
using MPI 
using ProgressMeter
using ClassicalSpinMC: Lattice, read_spin_configuration!, overwrite_keys!
using Logging
using Printf

function compute_magnetization(St::Array{Float64,2})::Array{Float64,2}
    # St is a N x t  matrix 
    N = Int(size(St)[1]//3)
    M = zeros(Float64, 3, size(St)[2])
    M[1,:] .= @views vec(sum(St[3 * collect(1:N) .- 2, :], dims=1))
    M[2,:] .= @views vec(sum(St[3 * collect(1:N) .- 1, :], dims=1))
    M[3,:] .= @views vec(sum(St[3 * collect(1:N),      :], dims=1))
    return M/N
end

"""
    run2DSpecSingle(lat::Lattice, ts::Array{Float64}, tau::Float64, B::H; kwargs...)::NTuple{3, Array{Float64,2}} where {H}

Run 2D spectroscopy for a single delay time tau. 

# Arguments:
- `lat::Lattice`: Lattice object containing spin configuration to time evolve
- `ts::Vector{Float64}`: time array 
- `B::Function`: time-dependent function that returns a magnetic field vector 
- `kwargs`: see `compute_St` documentation 
"""
function run2DSpecSingle(lat::Lattice, ts::Vector{Float64}, BA::F, BB::G, BAB::H; kwargs...) where {F,G,H}
    specA  = compute_St(ts, BA, lat; kwargs...)
    specB  = compute_St(ts, BB, lat; kwargs...)
    specAB  = compute_St(ts, BAB, lat; kwargs...)
    MA = compute_magnetization(specA)
    MB = compute_magnetization(specB)
    MAB = compute_magnetization(specAB)
    return MA, MB, MAB
end

function run2DSpecStack(stackfile::String, tmin::Float64, tmax::Float64, dt::Float64, taus::Vector{Float64}, BA::Function, BB::Function; 
                        override=false, report_interval=100, kwargs...)
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

    Ntau = length(taus)
    MA = zeros(Float64, 3, Ntau, Ntau)
    MB = copy(MA) 
    MAB = copy(MA)
    

    while length(file) > 0
        f = h5open(file, "r+") 
        exists = haskey(f, "spectroscopy")
        close(f)
        # if override and exists, skip the file 
        if exists && !override # read existing values from file 
            println("Exists; Skipping $file")
        else
            try 
                # initialize lattice by reading lattice metadata from params file
                lat = read_lattice_stack(file)
                read_spin_configuration!(lat,file) # read the spin configurations
                # do spectroscopy for each tau 
                println("Doing spectroscopy on $file")
                t0 = time()
                walltime = 0
                for (ind,tau) in enumerate(taus)
                    ts = collect(tmin:dt:tmax+tau)  
                    B(t) = BB(t-tau)
                    BAB(t) = BA(t) + B(t)

                    a,b,ab = run2DSpecSingle(lat, ts, BA, B, BAB; kwargs...)
                    MA[:,:,ind] .= a[:,(ts .>= tau)]
                    MB[:,:,ind] .= b[:,(ts .>= tau)]
                    MAB[:,:,ind] .= ab[:,(ts .>= tau)]

                    if ind % report_interval == 0
                        elapsed_time = time() - t0
                        walltime += elapsed_time
                        average_time = elapsed_time / report_interval 
                        estimated_remaining_time = average_time * Ntau - walltime 

                        str = ""
                        str *= "Rank $(rank+1)/$commSize: $file"
                        str *= "\t\tProgress: $ind/$(Nt-1)"
                        str *= @sprintf("\t\tTotal elapsed time : %.2f s \n", walltime)
                        str *= @sprintf("\t\tAverage time per tau : %.2f s \n", average_time)
                        str *= @sprintf("\t\tEstimated time remaining : %.2f s \n", estimated_remaining_time)
                        @info str 
                        t0 = time()
                    end 
                end
                walltime += time() - t0
                @info "\nCOMPLETED : total wall time $walltime s \n"

                # create a new group or overwrite existing group 
                println("Writing to $file on rank $rank")
                res = Dict{String, Any}("tmin"=>tmin, "tmax"=>tmax, "dt"=>dt, "taus"=>taus, 
                        "MA"=>MA, "MB"=>MB, "MAB"=>MAB)
                h5open(file, "r+") do f
                    if haskey(f, "spectroscopy")
                        g = f["spectroscopy"]
                        overwrite_keys!(g, res)
                    else
                        g = create_group(f, "spectroscopy")
                        for key in keys(res)
                            g[key] = res[key]
                        end
                    end
                end
                
            catch err 
                println("Something went wrong, pushing $file back to end of stack")
                pushToStack!(stackfile, file)
                rethrow(err)
            end
        end
        # pull next file from stack 
        file = pullFromStack!(stackfile) 
        attempts = 0 
        while length(file) == 0
            println("Stack empty; idling...")
            sleep(10)
            file = pullFromStack!(stackfile)
            attempts += 1
            if attempts > 3
                println("Stack empty; relinquishing job")
                break
            end 
        end
    end
end 

