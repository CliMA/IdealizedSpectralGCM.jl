export Atmo_Data, Compute_Abs_Vor!


struct Atmo_Data
    name::String


    nλ::Int64
    nθ::Int64
    nd::Int64
    
    do_mass_correction::Bool
    do_energy_correction::Bool
    do_water_correction::Bool
    use_virtual_temperature::Bool

    # physics related
    radius::Float64
    omega::Float64
    grav::Float64
    rdgas::Float64
    kappa::Float64
    rvgas::Float64
    cp_air::Float64

    #
    coriolis::Array{Float64,1}



end


function Atmo_Data(name::String,  
                   nλ::Int64, nθ::Int64, nd::Int64, 
                   do_mass_correction::Bool,
                   do_energy_correction::Bool,
                   do_water_correction::Bool,
                   use_virtual_temperature::Bool,
                   sinθ::Array{Float64,1},
                   radius::Float64, 
                   omega::Float64=7.292e-5, grav::Float64=9.80, rdgas::Float64=287.04, 
                   kappa::Float64=2.0/7.0, rvgas::Float64=461.50 )

    coriolis = 2 * omega * sinθ
    cp_air = rdgas/kappa
    

    Atmo_Data(name,  
    nλ, nθ, nd,
    do_mass_correction, do_energy_correction, do_water_correction,
    use_virtual_temperature,
    radius, omega, grav, rdgas, kappa, rvgas, cp_air,
    coriolis)     
end



function Compute_Abs_Vor!(grid_vor::Array{Float64,3}, coriolis::Array{Float64,1}, grid_absvor::Array{Float64,3})
    nλ, nθ, nd = size(grid_vor)

    for j=1:nθ
        grid_absvor[:,j,:] .= grid_vor[:,j,:] .+ coriolis[j]
    end
end