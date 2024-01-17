using FileWatching: watch_file
using Base.Filesystem
using ClassicalSpinMC: read_lattice

"""
Pushes file to end of stack 
"""
function pushToStack!(stackfile, file)
    # lockname = lock_file(stackfile) # lock file
    lock_file(stackfile) do (lh,lk)
        open(stackfile, "a") do io # write line to end stack 
            write(io, "$file\n")
        end
    end
end

"""
Reads first line from stackfile and creates new stackfile with first line removed 
"""
function pullFromStack!(stackfile::String)::String
    file = []
    sleep_rand(10) # sleep to prevent threads from accessing at the same time 
    lock_file(stackfile) do (lh,lk)
        (tmppath, tmpio) = mktemp(dirname(stackfile)) # make a temporary stackfile
        try
            open(stackfile) do io
                push!(file, readline(io)) # read and remove first line from io stream 
                for line in eachline(io, keep=true) # keep so the new line isn't chomped
                    write(tmpio, line)
                end
            end
            mv(tmppath, stackfile, force=true) # overwrite stackfile with first line removed if read successful 
            return file[1]
        finally
            close(tmpio)
        end
    end
end

"""
Sleep a random amount to prevent all of the threads from trying to access the stack at the same time 
"""
function sleep_rand(t::Real)
    wait = rand(Float64) * t # wait between 0 and 20 seconds 
    println("idling for $wait s")
    sleep(wait)
end

"""
Create a lock to a shared file.
"""
function lock_file(sharefilename::String)
    lockacquired = false
    lockfilename = sharefilename * ".lock"
    local lockfilehandle 
    while !lockacquired
        while isfile(lockfilename)
            # watch_file will notify if the file status changes, waiting until then
            # here we want to wait for the file to get deleted
            @info "$sharefilename locked, idling..."
            watch_file(lockfilename, 5.0) # timeout after 5 seconds 
        end
        try
            # try to acquire the lock by creating lock file with JL_O_EXCL (exclusive)
            lockfilehandle = Filesystem.open(lockfilename, JL_O_CREAT | JL_O_EXCL, 0o600)
            lockacquired = true
        catch err
            # in case the file was created between our `isfile` check above and the
            # `Filesystem.open` call, we'll get an IOError with error code `UV_EEXIST`.
            # In that case, we loop and try again. 
            if err isa Base.IOError && err.code == Base.UV_EEXIST
                @debug "Tried to create file when it already exists, looping again."
                continue
            else
                rethrow()
            end
        end
    end
    return lockfilehandle, lockfilename
end

function lock_file(f::Function, filename::String)
    io = lock_file(filename)
    try
        f(io)
    finally
        unlock_file(io...)
    end
end

"""
Frees up lock to shared file. 
"""
function unlock_file(lockfilehandle, lockfilename)
    # free up the lock so that the other process can acquire it if it needs
    try 
        close(lockfilehandle)
        Filesystem.unlink(lockfilename)
    catch err 
        @error "Failed to unlock file $filename"
        rethrow(err)
    end 

end

"""
Reads file from stack and returns lattice object (threadsafe).
"""
function read_lattice_stack(file::String)
    f = h5open(file, "r") 
    paramsfile = read(attributes(f)["paramsfile"])
    p_ = h5open(paramsfile, "r")
    try
        lat = read_lattice(p_) 
        return lat
    finally
        close(p_)
        close(f)
    end
end