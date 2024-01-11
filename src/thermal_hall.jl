using BinningAnalysis
using ProgressMeter
using HDF5
using ClassicalSpinMC: overwrite_keys!

function compute_site_currents(path::String, dest::String)
    println("Initializing LogBinner in $path")
    files = readdir(path)
    f0 = h5open(string(path, files[1]), "r")
    shape = size(read(f0["transport/ji"]))
    close(f0)

    # initialize currents
    ji = LogBinner(zeros(Float64, shape...))

    # collecting averages
    count = 0
    println("Collecting averages from ", length(files), " files")
    @showprogress for file in files 
        h5open(string(path, file), "r") do f 
            try 
                push!(ji, read(f["transport/ji"]))
                count += 1
            catch
            end
        end
    end

    site_current = mean(ji) 
    # write to configuration file 
    println("Writing to $dest")
    d = h5open(dest, "r+")
    res = Dict("ji"=>site_current)
    g = haskey(d, "transport") ? d["transport"] : create_group(d, "transport")
    overwrite_keys!(g, res)
    close(d)

    println("Successfully averaged $count/", length(files)," files")
end


# function compute_thermal_hall(path::String, dest::String, params::Dict{String, Float64})
#     # initialize LogBinner
#     println("Initializing LogBinner in $path")
#     files = readdir(path)
#     f0 = h5open(string(path, files[1]), "r")
#     T = read(attributes(f0)["T"])
#     t = read(f0["transport/t"])

#     shape = size(read(f0["transport/jxjy"]))
#     close(f0)

#     # initialize correlation functions
#     Cxy = LogBinner(zeros(Float64, shape...))
#     Cyx = LogBinner(zeros(Float64, shape...))

#     # initialize energy magnetization 
#     EMxy = LogBinner()
#     EMyx = LogBinner()

#     # collecting averages
#     println("Collecting averages from ", length(files), " files")
#     @showprogress for file in files 
#         h5open(string(path, file), "r") do f 
#             push!(Cxy, read(f["transport/jxjy"]))
#             push!(Cyx, read(f["transport/jyjx"]))
#             push!(EMxy, read(f["transport/jExy"]))
#             push!(EMyx, read(f["transport/jEyx"]))
#         end
#     end

#     cxy = mean(Cxy) 
#     cyx = mean(Cyx)

#     dt = params["dt"]

#     # compute thermal Hall conductivity 
#     k_xy_0 = 1/T^2 * sum(mean(cxy)) * dt 
#     k_yx_0 = 1/T^2 * sum(mean(cyx)) * dt 
#     k_xy_1 = mean(EMxy) / T                 # these already multiplied by -2 in transport.jl
#     k_yx_1 = mean(EMyx) / T

#     # write to configuration file 
#     println("Writing to $dest")
#     d = h5open(dest, "r+")
#     res = Dict("k_xy_0"=>k_xy_0, "k_xy_1"=>k_xy_1, "k_yx_0"=>k_yx_0, "k_yx_1"=>k_yx_1)
#     g = haskey(d, "thermal_hall") ? d["thermal_hall"] : create_group(d, "thermal_hall")
#     overwrite_keys!(g, res)
    
#     g2 = haskey(d, "transport") ? d["transport"] : create_group(d, "transport")
#     overwrite_keys!(g2, Dict("t"=>t, "Cxy"=>cxy, "Cyx"=>cyx))
#     overwrite_keys!(g2, params)
#     close(d)

#     println("Successfully averaged ",length(files), " files")
# end

function compute_thermal_hall(path::String, dest::String, params::Dict{String, Float64})
    # initialize LogBinner
    println("Initializing LogBinner in $path")
    files = readdir(path)
    # find the first instance of a file with the transport group and initialize variables 
    init_var = Dict{String, Any}()
    transport_exists = false
    for file in files
        f = h5open(string(path, file), "r")
        if haskey(f, "transport")
            init_var["T"] = read(attributes(f)["T"])
            init_var["t"] = read(f["transport/t"])
            init_var["shape"] = size(read(f["transport/jxjy"]))
            close(f)
            transport_exists = true
            break
        end
        close(f)
    end
    !transport_exists && throw("No files with transport group found in $path")

    T = init_var["T"]
    t = init_var["t"]
    shape = init_var["shape"]
    println(shape)

    # initialize correlation functions
    Cxy = LogBinner(zeros(Float64, shape...))
    Cyx = LogBinner(zeros(Float64, shape...))

    # initialize energy magnetization 
    EMxy = LogBinner()
    EMyx = LogBinner()

    count = 0
    # collecting averages
    @showprogress for file in files 
        h5open(string(path, file), "r") do f
            if haskey(f, "transport")
                try
                    push!(Cxy, read(f["transport/jxjy"]))
                    push!(Cyx, read(f["transport/jyjx"]))
                    push!(EMxy, read(f["transport/jExy"]))
                    push!(EMyx, read(f["transport/jEyx"]))
                    count += 1 
                catch
                    # println("Failed on file $file")
                end

            end
        end
    end

    cxy = mean(Cxy) 
    cyx = mean(Cyx)

    dt = params["dt"]

    # compute thermal Hall conductivity 
    k_xy_0 = 1/T^2 * sum(mean(cxy)) * dt 
    k_yx_0 = 1/T^2 * sum(mean(cyx)) * dt 
    k_xy_1 = mean(EMxy) / T                 # these already multiplied by -2 in transport.jl
    k_yx_1 = mean(EMyx) / T

    # write to configuration file 
    println("Writing to $dest")
    d = h5open(dest, "r+")
    res = Dict("k_xy_0"=>k_xy_0, "k_xy_1"=>k_xy_1, "k_yx_0"=>k_yx_0, "k_yx_1"=>k_yx_1)
    g = haskey(d, "thermal_hall") ? d["thermal_hall"] : create_group(d, "thermal_hall")
    overwrite_keys!(g, res)
    
    g2 = haskey(d, "transport") ? d["transport"] : create_group(d, "transport")
    overwrite_keys!(g2, Dict("t"=>t, "Cxy"=>cxy, "Cyx"=>cyx))
    overwrite_keys!(g2, params)
    close(d)

    println("Successfully averaged $count/",length(files), " files")
end



