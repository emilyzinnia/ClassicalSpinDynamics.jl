# ClassicalSpinDynamics.jl

This package is for simulating various dynamical quantities of classical spin systems. It works off of the files generated from the ClassicalSpinMC.jl package. Note that this package is under development and will experience frequent updates! 

## Prerequisites 
* `OpenMPI` or `IntelMPI`. The package uses `MPI.jl`, so on a cluster you may need to configure your MPI Julia installation to use the system-provided MPI backend. See the [MPI.jl documentation](https://juliaparallel.org/MPI.jl/stable/configuration/) for details. 
* `HDF5`
* `DifferentialEquations.jl` for controlling the solver algorithm used in solving the ODEs.


## Installation 
1. In your desired installation directory, clone the github repository. 
2. Launch a Julia REPL and type the following command: 

`using Pkg; Pkg.add("$INSTALLATION_PATH/ClassicalSpinDynamics")`

where `$INSTALLATION_PATH` is the path to the package repository. 
