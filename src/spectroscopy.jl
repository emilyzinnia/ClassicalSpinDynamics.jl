using HDF5
using MPI 
using ProgressMeter
using ClassicalSpinMC: Lattice, read_spin_configuration!

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

function run2DSpecStack(stackfile::String, ts::Vector{Float64}, taus::Vector{Float64}, BA::Function, BB::Function; 
                        override=false, kwargs...)
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
    
    # initialize lattice by reading lattice metadata from params file
    lat = read_lattice_stack(file)

    MA = zeros(Float64, 3, size(ts)[1], size(taus)[1])
    MB = copy(MA) 
    MAB = copy(MA)
    Nt = size(taus)[1]+1

    while length(file) > 0
        try
            # if override and exists, skip the file 
            f = h5open(file, "r+") 
            exists = haskey(f, "spectroscopy")
            close(f)
            if exists && !override # read existing values from file 
                println("Exits; Skipping $file")
                continue 
            else
                read_spin_configuration!(file, lat) # read the spin configurations
                # do spectroscopy for each tau 
                println("Doing 2D spectroscopy on $file")
                @showprogress for (ind,tau) in enumerate(taus)
                    a,b,ab = run2DSpecSingle(lat, ts, tau, BA, BB; kwargs...)
                    MA[:,:,ind] .= a
                    MB[:,Nt:end,ind] .= b
                    MAB[:,:,ind] .= ab
                end
    
                # create a new group or overwrite existing group 
                println("Writing to $file")
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
            println(err.msg)
            println("Something went wrong, pushing $file back to end of stack")
            pushToStack!(stackfile, file)
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

