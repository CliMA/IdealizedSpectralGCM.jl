__precompile__(true)
module JGCM

using Revise
using LinearAlgebra
using FFTW
using Statistics 
using JLD2
using MAT
import PyPlot

include("Atmos_Spectral/Gauss_And_Legendre.jl")
include("Atmos_Spectral/Spectral_Spherical_Mesh.jl")
include("Atmos_Spectral/Atmo_Data.jl")
include("Atmos_Spectral/Dyn_Data.jl")
include("Atmos_Spectral/Vert_Coordinate.jl")
include("Atmos_Spectral/Time_Integrator.jl")

include("Atmos_Spectral/Semi_Implicit.jl")
include("Atmos_Spectral/Press_And_Geopot.jl")
include("Atmos_Spectral/Output_Manager.jl")
# Params
include("Atmos_Param/HS_Forcing.jl")


# Applications
include("Atmos_Spectral/Barotropic_Dynamics.jl")
include("Atmos_Spectral/Shallow_Water_Dynamics.jl")
include("Atmos_Spectral/Spectral_Dynamics.jl")
end