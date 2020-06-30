export Shallow_Water_Physics!, Implicit_Correction!, Shallow_Water_Dynamics!
#################
# Dyn_Data does not has h 
# we use grid_δlnps, grid_ps spe_lnps as 
#        grid_δh     grid_h  spe_h
##################
function Shallow_Water_Physics!(dyn_data::Dyn_Data, kappa_m::Float64, 
  kappa_t::Float64, h_eq::Array{Float64,3})
  
  grid_δu, grid_δv, grid_δh = dyn_data.grid_δu, dyn_data.grid_δv, dyn_data.grid_δlnps
  grid_u, grid_v, grid_h = dyn_data.grid_u_c, dyn_data.grid_v_c, dyn_data.grid_ps_c
  # grid_δu .= 0.0
  # grid_δv .= 0.0
  # grid_δh .= 0.0

  grid_δu  .= -kappa_m * grid_u
  grid_δv  .= -kappa_m * grid_v
  grid_δh .= -kappa_t * (grid_h - h_eq)
  
  
end



function Implicit_Correction!(integrator::Filtered_Leapfrog, 
  spe_div_c::Array{ComplexF64,3}, spe_div_p::Array{ComplexF64,3}, 
  spe_h_c::Array{ComplexF64,3}, spe_h_p::Array{ComplexF64,3}, h_0::Float64,  
  spe_δdiv::Array{ComplexF64,3}, spe_δh::Array{ComplexF64,3})
  """
  See Shallow_Water_Dynamics!
  """
  
  implicit_coef = integrator.implicit_coef
  eigen = integrator.laplacian_eigen
  init_step = integrator.init_step
  if init_step
    Δt = integrator.Δt
  else
    Δt = 2.0 * integrator.Δt
  end

  #for the first time step spe_h_c = spe_h_p and spe_div_c = spe_div_p
  spe_δdiv  .+=  eigen .* (spe_h_c - spe_h_p)
  spe_δh   .+=  h_0 * (spe_div_c - spe_div_p)
  
  μ = implicit_coef * Δt
  μ2 = μ^2
  
  spe_δdiv .= (spe_δdiv .- μ * eigen .* spe_δh) ./ (1.0 .- μ2 * eigen * h_0)
  
  spe_δh  .-=  μ * h_0 * spe_δdiv

  
end

"""
The governing equations are
∂vor/∂t = ∇ ((f + vor) V)  := δvor
∂div/∂t = ∇ × ((f + vor) V) - ∇^2(E + H) := δdiv
∂H/∂t = -H div - ∇H V  := δH

The implicit scheme treat the -∇^2 H and -H0 div terms in the second and third equations implicitly

(vor(i+1) - vor(i-1))/2Δt = δvor(i) 
(div(i+1) - div(i-1))/2Δt = δdiv(i) + ∇^2H(i) - ∇^2(αH(i+1) + (1-α)H(i-1))
(H(i+1) - H(i-1))/2Δt = δH(i) + H0div(i) - H0(αdiv(i+1) + (1-α)div(i-1))

(vor(i+1) - vor(i-1))/2Δt = δvor(i) 
(div(i+1) - div(i-1))/2Δt = δdiv(i) + ∇^2(H(i)-H(i-1)) - α∇^2(H(i+1) - H(i-1))
(H(i+1) - H(i-1))/2Δt = δH(i) + H0(div(i) -div(i-1)) - αH0(div(i+1) - div(i-1))


(vor(i+1) - vor(i-1))/2Δt = δvor(i) 
(div(i+1) - div(i-1))/2Δt = δdiv(i) + ∇^2(H(i)-H(i-1)) - α∇^2(H(i+1) - H(i-1))
(H(i+1) - H(i-1))/2Δt = δH(i) + H0(div(i) -div(i-1)) - αH0(div(i+1) - div(i-1))

Let δvor = (vor(i+1) - vor(i-1))/2Δt, δdiv = (div(i+1) - div(i-1))/2Δt and δH = (H(i+1) - H(i-1))/2Δt
We have Implicit_Correction!

δdiv = δdiv(i) + ∇^2(H(i)-H(i-1)) - 2Δtα∇^2 δH
δH = δH(i) + H0(div(i) -div(i-1)) - 2ΔtαH0 δdiv 

We have 
δdiv = δdiv(i) + ∇^2(H(i)-H(i-1)) - 2Δtα∇^2 (δH(i) + H0(div(i) -div(i-1)) - 2ΔtαH0 δdiv )

δdiv + - 2Δtα2ΔtαH0∇^2 δdiv = δdiv(i) + ∇^2(H(i)-H(i-1)) - 2Δtα∇^2 (δH(i) + H0(div(i) -div(i-1)))
"""
function Shallow_Water_Dynamics!(mesh::Spectral_Spherical_Mesh, 
  atmo_data::Atmo_Data, h_0::Float64, dyn_data::Dyn_Data, 
  integrator::Filtered_Leapfrog)
  # n: next, c: current, p: previous
  
  nλ, nθ = mesh.nλ, mesh.nθ
  
  spe_vor_n = dyn_data.spe_vor_n
  spe_vor_c = dyn_data.spe_vor_c
  spe_vor_p = dyn_data.spe_vor_p
  spe_δvor = dyn_data.spe_δvor
  grid_vor = dyn_data.grid_vor
  
  
  
  spe_div_n = dyn_data.spe_div_n
  spe_div_c = dyn_data.spe_div_c
  spe_div_p = dyn_data.spe_div_p
  spe_δdiv = dyn_data.spe_δdiv
  grid_div = dyn_data.grid_div
  
  
  spe_h_n = dyn_data.spe_lnps_n
  spe_h_c = dyn_data.spe_lnps_c
  spe_h_p = dyn_data.spe_lnps_p
  spe_δh = dyn_data.spe_δlnps
  grid_h_n = dyn_data.grid_ps_n
  grid_h = dyn_data.grid_ps_c
  grid_δh = dyn_data.grid_δlnps
  
  grid_u_n = dyn_data.grid_u_n
  grid_u = dyn_data.grid_u_c
  grid_δu = dyn_data.grid_δu
  grid_v_n = dyn_data.grid_v_n
  grid_v = dyn_data.grid_v_c
  grid_δv = dyn_data.grid_δv
  
  grid_absvor = dyn_data.grid_absvor
  Compute_Abs_Vor!(grid_vor, atmo_data.coriolis, grid_absvor)
  
  # compute explicit residual
  grid_δu .+=   grid_absvor .* grid_v
  grid_δv .+=  -grid_absvor .* grid_u
  
  # vorticity
  Vor_Div_From_Grid_UV!(mesh, grid_δu, grid_δv, spe_δvor, spe_δdiv)

  
  # divergence
  grid_kin, spe_kin = dyn_data.grid_d_full1, dyn_data.spe_d1
  spe_energy = dyn_data.spe_energy
  grid_kin .= 0.5 * (grid_u.^2 + grid_v.^2)
  Trans_Grid_To_Spherical!(mesh, grid_kin, spe_kin)
  spe_energy .= spe_kin + spe_h_c
  Apply_Laplacian!(mesh, spe_energy)
  spe_δdiv .-= spe_energy

  
  # heights
  Add_Horizontal_Advection!(mesh, spe_h_n, grid_u, grid_v, grid_δh)
  grid_δh .-= grid_h .* grid_div
  Trans_Grid_To_Spherical!(mesh, grid_δh, spe_δh)

  
  
  Implicit_Correction!(integrator, spe_div_c, spe_div_p, 
  spe_h_c, spe_h_p, h_0,  
  spe_δdiv, spe_δh)

  
  
  Compute_Spectral_Damping!(integrator, spe_vor_c, spe_vor_p, spe_δvor)
  Compute_Spectral_Damping!(integrator, spe_div_c, spe_div_p, spe_δdiv)
  Compute_Spectral_Damping!(integrator, spe_h_c, spe_h_p, spe_δh)

  Filtered_Leapfrog!(integrator, spe_δvor, spe_vor_p, spe_vor_c, spe_vor_n)
  Filtered_Leapfrog!(integrator, spe_δdiv, spe_div_p, spe_div_c, spe_div_n)
  Filtered_Leapfrog!(integrator, spe_δh, spe_h_p, spe_h_c, spe_h_n)

  
  # update grid_u, grid_v, grid_vor, grid_div, grid_h
  Trans_Spherical_To_Grid!(mesh, spe_vor_n, grid_vor)
  Trans_Spherical_To_Grid!(mesh, spe_div_n, grid_div)
  Trans_Spherical_To_Grid!(mesh, spe_h_n, grid_h_n)
  UV_Grid_From_Vor_Div!(mesh, spe_vor_n, spe_div_n, grid_u_n, grid_v_n)
  
  @show norm(grid_u_n), norm(grid_v_n), norm(grid_h_n)
  
  # update data in to the next step
  Time_Advance!(dyn_data)
  
end




#Shallow_Water_Main()
