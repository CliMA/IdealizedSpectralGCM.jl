export Filtered_Leapfrog, Update_Init_Step!, Get_Δt, Get_ξ, Compute_Spectral_Damping!, Filtered_Leapfrog!
mutable struct Filtered_Leapfrog
    robert_coef::Float64

    damping_order::Int64
    damping_coef::Float64
    damping::Array{Float64,2}
    laplacian_eigen::Array{Float64,2}

    implicit_coef::Float64

    Δt::Int64
    init_step::Bool
    time::Int64
    start_time::Int64
    end_time::Int64

end

function Filtered_Leapfrog(robert_coef::Float64, 
                           damping_order::Int64, damping_coef::Float64, eigen::Array{Float64,2},
                           implicit_coef::Float64,
                           Δt::Int64, init_step::Bool, start_time::Int64, end_time::Int64)
    @assert(damping_order%2 == 0)
    
    num_fourier, num_spherical = size(eigen) .- 1 
    #resolution_independent damping
    #damping = damping_coef*(eigen).^damping_order
    #resolution_dependent damping

    damping = damping_coef*(eigen/eigen[1,num_spherical]).^damping_order


    laplacian_eigen = eigen
    time = start_time
    Filtered_Leapfrog(robert_coef, damping_order, damping_coef, damping, laplacian_eigen, 
    implicit_coef, Δt, init_step, time, start_time, end_time)
end

function Update_Init_Step!(integrator::Filtered_Leapfrog)
    @assert(integrator.init_step)
    integrator.init_step = false
end


function Get_Δt(integrator::Filtered_Leapfrog)
    init_step = integrator.init_step
    Δt = integrator.Δt
    return (init_step ? Δt : 2*Δt) 
end

function Get_ξ(integrator::Filtered_Leapfrog)
    init_step = integrator.init_step
    Δt, implicit_coef = integrator.Δt, integrator.implicit_coef
    return (init_step ? Δt*implicit_coef : 2*Δt*implicit_coef) 
end

function Compute_Spectral_Damping!(integrator::Filtered_Leapfrog,
                                   Qc::Array{ComplexF64,3}, Qp::Array{ComplexF64,3}, 
                                   δQ::Array{ComplexF64,3})
    """
    update δQ

    (Q^{i+1} - Q^{i-1})/2Δt = dQ - ν(-1)^n ∇^2n Q^{i+1}
                            = dQ - ν(-1)^n ∇^2n(Q^{i+1} -  Q^{i-1}) - ν(-1)^n ∇^2n Q^{i-1}

    (Q^{i+1} - Q^{i-1})(1/2Δt +  ν |σ|^n) = dQ - ν|σ|^n  Q^{i-1}

    (Q^{i+1} - Q^{i-1})/2Δt =  = (dQ - ν|σ|^n Q^{i-1}) /(1 +  ν 2Δt |σ|^n)

    """
    init_step = integrator.init_step
    damping = integrator.damping
    Δt = integrator.Δt
      
    if (init_step) 
        δQ .= (δQ - Qc .* damping)./(1.0 .+ Δt*damping)
    else
        δQ .= (δQ - Qp .* damping)./(1.0 .+ 2Δt*damping)
    end
end




function Filtered_Leapfrog!(integrator::Filtered_Leapfrog,
                   δQ::Array{ComplexF64,3},
                   Qp::Array{ComplexF64,3}, Qc::Array{ComplexF64,3}, Qn::Array{ComplexF64,3})

    init_step = integrator.init_step
    robert_coef = integrator.robert_coef
    Δt = integrator.Δt
    if (init_step) 
        Qn .= Qc + Δt*δQ
        Qc .+= robert_coef*(-1.0*Qc + Qn)
    else
        Qc .+= robert_coef*(Qp - 2*Qc)
        Qn .= Qp + 2*Δt*δQ
        Qc .+= robert_coef*Qn
    end
end

