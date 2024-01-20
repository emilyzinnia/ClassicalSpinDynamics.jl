using HDF5
using MPI 
using ProgressMeter
using ClassicalSpinMC: Lattice, read_spin_configuration!, overwrite_keys!
using Logging, LoggingExtras
using Printf

function compute_magnetization(St::Array{Float64,2})::Array{Float64,2}
    # St is a t x N matrix 
    N = Int(size(St)[2]//3)
    M = zeros(Float64, 3, size(St)[1])
    M[1,:] .= sum(St[:, 3 * collect(1:N) .- 2], dims=2)
    M[2,:] .= sum(St[:, 3 * collect(1:N) .- 1], dims=2)
    M[3,:] .= sum(St[:, 3 * collect(1:N)     ], dims=2)
    return M/N
end

"""
    run2DSpecSingle(lat::Lattice, ts::Array{Float64}, tau::Float64, B::H; kwargs...)::NTuple{3, Array{Float64,2}} where {H}

Run 2D spectroscopy for a single delay time tau. 

# Arguments:
- `lat::Lattice`: Lattice object containing spin configuration to time evolve
- `ts::Vector{Float64}`: time array 
- `tau::Float64`: delay time 
- `B::Function`: time-dependent function that returns a magnetic field vector 
- `kwargs`: see `compute_St` documentation 
"""
function run2DSpecSingle(lat::Lattice, ts::Vector{Float64}, tau::Float64, BA::Function, BB::Function; kwargs...)::NTuple{3, Array{Float64,2}}
    specA  = compute_St(ts, BA, lat; kwargs...)
    specB  = compute_St(ts[ts .>= 0.0], BB, lat; kwargs...)
    specAB  = compute_St(ts, BA, BB, lat; kwargs...)
    MA = compute_magnetization(specA)
    MB = compute_magnetization(specB)
    MAB = compute_magnetization(specAB)
    return MA, MB, MAB
end

function run2DSpecStack(stackfile::String, ts::Vector{Float64}, taus::Vector{Float64}, B::Function, BB::Function; 
                        override=false, debug=false, report_interval=100, kwargs...)
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

    MA = zeros(Float64, 3, size(ts)[1], size(taus)[1])
    MB = copy(MA) 
    MAB = copy(MA)
    Nt = size(taus)[1]+1

    # initialize logging 
    logfilepath = dirname(file) * "/logs"
    try 
        rank == 0 && !isfile(logfilepath) && mkdir(logfilepath)
    catch err
        # If multiple threads attempt to create the file at the same time and it already exists, skip 
        if !(err isa Base.IOError && err.code == Base.UV_EEXIST)
            rethrow(err)
        end
    end 

    while length(file) > 0
        f = h5open(file, "r+") 
        exists = haskey(f, "spectroscopy")
        close(f)
        # if override and exists, skip the file 
        if exists && !override # read existing values from file 
            println("Exists; Skipping $file")
        else
            # create a new logfile
            logfilename = logfilepath * "/" * basename(file) * ".log"
            logio = open(logfilename, "w")
            logger = debug ? ConsoleLogger(stderr, Logging.Debug) : FileLogger(logio)
            try 
                # initialize lattice by reading lattice metadata from params file
                with_logger(logger) do 
                    lat = read_lattice_stack(file)
                    read_spin_configuration!(lat,file) # read the spin configurations
                    # do spectroscopy for each tau 
                    println("Doing spectroscopy on $file")
                    t0 = time()
                    walltime = 0
                    for (ind,tau) in enumerate(taus)
                        BA(t) = B(t, tau)
                        a,b,ab = run2DSpecSingle(lat, ts, tau, BA, BB; kwargs...)
                        MA[:,:,ind] .= a
                        MB[:,Nt:end,ind] .= b
                        MAB[:,:,ind] .= ab

                        if ind % report_interval == 0
                            elapsed_time = time() - t0
                            walltime += elapsed_time
                            average_time = elapsed_time / report_interval 
                            estimated_remaining_time = average_time * (Nt-1) - walltime 

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
                    res = Dict{String, Any}("ts"=>ts, "taus"=>taus, "MA"=>MA, "MB"=>MB, "MAB"=>MAB)
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
                end 
            catch err 
                println("Something went wrong, pushing $file back to end of stack")
                pushToStack!(stackfile, file)
                rethrow(err)
            finally
                close(logio)
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

