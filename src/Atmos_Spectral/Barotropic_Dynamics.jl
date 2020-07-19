export Barotropic_Dynamics!
function Barotropic_Dynamics!(mesh::Spectral_Spherical_Mesh, 
                              atmo_data::Atmo_Data, dyn_data::Dyn_Data, 
                              integrator::Filtered_Leapfrog)
  # n: next, c: current, p: previous
  
    nλ, nθ = mesh.nλ, mesh.nθ
  
    spe_vor_n = dyn_data.spe_vor_n
    spe_vor_c = dyn_data.spe_vor_c
    spe_vor_p = dyn_data.spe_vor_p
  
    grid_u, grid_u_n = dyn_data.grid_u_c, dyn_data.grid_u_n
    grid_δu = dyn_data.grid_δu
    grid_v, grid_v_n = dyn_data.grid_v_c, dyn_data.grid_v_n
    grid_δv = dyn_data.grid_δv
    grid_vor = dyn_data.grid_vor 
  
    spe_δvor = dyn_data.spe_δvor
    spe_δdiv = dyn_data.spe_δdiv
  
    spe_zeros = dyn_data.spe_zeros
  
    grid_absvor =  dyn_data.grid_absvor

    Compute_Abs_Vor!(grid_vor, atmo_data.coriolis, grid_absvor)
  
    grid_δu .=   grid_absvor .* grid_v
    grid_δv .=  -grid_absvor .* grid_u
  
    Vor_Div_From_Grid_UV!(mesh, grid_δu, grid_δv, spe_δvor, spe_δdiv)
  
  
    Compute_Spectral_Damping!(integrator,  spe_vor_c, spe_vor_p, spe_δvor)
  
    Filtered_Leapfrog!(integrator, spe_δvor, spe_vor_p, spe_vor_c, spe_vor_n)
  
    Trans_Spherical_To_Grid!(mesh, spe_vor_n, grid_vor)
  
    UV_Grid_From_Vor_Div!(mesh, spe_vor_n,  spe_zeros, grid_u_n, grid_v_n)

    @show norm(grid_u_n), norm(grid_v_n), norm(grid_vor)
  
    Time_Advance!(dyn_data)
  
end





