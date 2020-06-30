using JGCM


function Shallow_Water_Main()
    # the decay of a sinusoidal disturbance to a zonally symmetric flow 
    # that resembles that found in the upper troposphere in Northern winter.
    name = "Shallow_Water"
    num_fourier, nθ, nd = 85, 128, 1
    num_spherical = num_fourier + 1
    nλ = 2nθ
    
    radius = 6371.0e3
    omega = 7.292e-5
    day_to_sec = 86400
    fric_damp_time  = 20.0 * day_to_sec
    therm_damp_time = 10.0 * day_to_sec
    h_0             = 3.e04
    h_amp           = 2.e04
    h_lon           =  90.0 * pi / 180
    h_lat           =  25.0 * pi / 180
    h_width         =  15.0 * pi / 180
    h_itcz          = 1.e05
    itcz_width      =  4.0 * pi / 180
    kappa_m = 1.0 / fric_damp_time
    kappa_t = 1.0 / therm_damp_time
    
    # Initialize mesh
    mesh = Spectral_Spherical_Mesh(num_fourier, num_spherical, nθ, nλ, nd, radius)
    θc, λc = mesh.θc,  mesh.λc
    cosθ, sinθ = mesh.cosθ, mesh.sinθ
    
    
  
    # Initialize atmo_data
    atmo_data = Atmo_Data(name, nλ, nθ, nd, false, false, false, false, sinθ, radius, omega)
    
    # Initialize integrator
    damping_order = 4
    damping_coef = 1.e-04
    robert_coef  = 0.04 
    
    implicit_coef = 0.5
    
    start_time = 0
    end_time = 691200 #
    Δt = 1200
    init_step = true
    
    integrator = Filtered_Leapfrog(robert_coef, 
    damping_order, damping_coef, mesh.laplacian_eig,
    implicit_coef,
    Δt, init_step, start_time, end_time)
    
    # Initialize data
    dyn_data = Dyn_Data(name, num_fourier, num_spherical, nλ, nθ, nd)
    
    grid_u, grid_v = dyn_data.grid_u_c, dyn_data.grid_v_c
    grid_u .= 0.0
    grid_v .= 0.0 
    
    spe_vor_c, spe_div_c = dyn_data.spe_vor_c, dyn_data.spe_div_c
    Vor_Div_From_Grid_UV!(mesh, grid_u, grid_v, spe_vor_c, spe_div_c) 
    Trans_Spherical_To_Grid!(mesh, spe_vor_c,  dyn_data.grid_vor)
    Trans_Spherical_To_Grid!(mesh, spe_div_c,  dyn_data.grid_div)
    
    grid_h, spe_h_c = dyn_data.grid_lnps, dyn_data.spe_lnps_c
    grid_h .= h_0
    Trans_Grid_To_Spherical!(mesh, grid_h, spe_h_c)
    
    
    
    h_eq = zeros(Float64, nλ, nθ, 1)
    for j = 1:nθ
      d2 = (θc[j] / itcz_width)^2
      for i = 1:nλ
        x2 = ((λc[i] - h_lon) / (2.0 * h_width))^2
        y2 = ((θc[j] - h_lat) / h_width)^2
        h_eq[i,j, 1] = h_0 + h_amp * exp(-(x2 + y2)) + h_itcz * exp(-d2)
      end
    end
    
  
    NT =  Int64(end_time / Δt)
    time = start_time
  
    Shallow_Water_Physics!(dyn_data, kappa_m, kappa_t, h_eq)
    Shallow_Water_Dynamics!(mesh, atmo_data, h_0, dyn_data, integrator)
    Update_Init_Step!(integrator)
    time += Δt 
    for i = 2:NT
      Shallow_Water_Physics!(dyn_data, kappa_m, kappa_t, h_eq)
      Shallow_Water_Dynamics!(mesh, atmo_data, h_0, dyn_data, integrator)
      time += Δt
      @info time
    end
  
    Lat_Lon_Pcolormesh(mesh, grid_h,  1, "Shallow_Water_h.png")
    
    
  end