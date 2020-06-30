export Compute_Pressures_And_Heights!, Half_Level_Pressures!, Pressure_Variables!, Compute_Geopotential!

function Compute_Pressures_And_Heights!(atmo_data::Atmo_Data, vert_coord::Vert_Coordinate,     
                                      grid_ps::Array{Float64,3}, grid_geopots::Array{Float64,3}, grid_t::Array{Float64,3}, 
                                      grid_p_half::Array{Float64,3},  grid_Δp::Array{Float64,3},
                                      grid_lnp_half::Array{Float64,3}, grid_p_full::Array{Float64,3}, grid_lnp_full::Array{Float64,3},
                                      grid_z_full::Array{Float64,3}, grid_z_half::Array{Float64,3})

  grav = atmo_data.grav

  Pressure_Variables!(vert_coord, grid_ps, grid_p_half, grid_Δp,
                       grid_lnp_half, grid_p_full, grid_lnp_full)

  Compute_Geopotential!(vert_coord,atmo_data, grid_lnp_half, grid_lnp_full,  grid_t, grid_geopots, grid_z_full, grid_z_half)

  grid_z_full ./= grav
  grid_z_half ./= grav

end 



function Half_Level_Pressures!(vert_coord::Vert_Coordinate, grid_ps::Array{Float64,3}, grid_p_half::Array{Float64,3}) 
  
  nd = vert_coord.nd
  bk = vert_coord.bk
  ak = vert_coord.ak
  
  # pk = ak * pref
  for k=1:nd+1
    grid_p_half[:,:,k] .= ak[k] .+ bk[k]*grid_ps[:,:,1]
  end 
end 


function Pressure_Variables!(vert_coord::Vert_Coordinate, grid_ps::Array{Float64,3}, grid_p_half::Array{Float64,3}, grid_Δp::Array{Float64,3},
  grid_lnp_half::Array{Float64,3}, grid_p_full::Array{Float64,3}, grid_lnp_full::Array{Float64,3})
  
  @assert(size(grid_ps)[3] == 1)
  
  Half_Level_Pressures!(vert_coord, grid_ps, grid_p_half)
  nd = vert_coord.nd
  grid_Δp .= grid_p_half[:,:,2:nd+1] - grid_p_half[:,:,1:nd]
  zero_top = vert_coord.zero_top
  
  if (vert_coord.vert_difference_option == "simmons_and_burridge") 
    
    k_top = (zero_top ? 2 : 1) 
    
    grid_lnp_half[:,:,k_top:nd+1] .= log.(grid_p_half[:,:,k_top:nd+1])
    
    #lnp_{k} = (p_{k+1/2}lnp_{k+1/2} - p_{k-1/2}lnp_{k-1/2})/Δp_k - 1
    #        = [(p_{k+1/2}-p_{k-1/2})lnp_{k+1/2} + p_{k-1/2}(lnp_{k+1/2} - lnp_{k-1/2})]/Δp_k - 1
    #        = lnp_{k+1/2} + [p_{k-1/2}(lnp_{k+1/2} - lnp_{k-1/2})]/Δp_k - 1
    grid_lnp_full[:,:,k_top:nd] .= grid_lnp_half[:,:,k_top+1:nd+1] .+ grid_p_half[:,:,k_top:nd].*(grid_lnp_half[:,:,k_top+1:nd+1] - grid_lnp_half[:,:,k_top:nd])./grid_Δp[:,:,k_top:nd] .- 1.0 
    
    if (zero_top) 
      grid_lnp_half[:,:,1] .= 0.0
      grid_lnp_full[:,:,1] .= grid_lnp_half[:,:,2] .- 1.0 
    end
  
  else
    error("vert_difference_option ",vert_coord.vert_difference_option, " is not a valid value for option")
    
  end
  grid_p_full .= exp.(grid_lnp_full)
end



function Compute_Geopotential!(vert_coord::Vert_Coordinate, atmo_data::Atmo_Data, 
  grid_lnp_half::Array{Float64, 3}, grid_lnp_full::Array{Float64, 3},  
  grid_t::Array{Float64, 3}, 
  grid_geopots::Array{Float64, 3}, grid_geopot_full::Array{Float64, 3}, grid_geopot_half::Array{Float64, 3})
  
  use_virtual_temperature = atmo_data.use_virtual_temperature
  rvgas, rdgas = atmo_data.rvgas, atmo_data.rdgas
  zero_top = vert_coord.zero_top
  nd = vert_coord.nd


  grid_geopot_half[:,:,nd+1] .= grid_geopots[:,:,1]
  
  if zero_top  #todo (pk(1).eq.0.0) then
    k_top = 2
    grid_geopot_half[:,:,1] .= 0.0
  else
    k_top = 1
  end
  
  # if (use_virtual_temperature) 
  #   virtual_t = grid_t .* (1. + (rvgas/rdgas - 1.)*grid_q)
  # else
  #   virtual_t = grid_t
  # end


  virtual_t = grid_t
  
  
  for k=nd:-1:k_top
    #Φ_{k-1/2} = Φ_{k+1/2} + RT_k(ln p_{k+1/2} - ln p_{k-1})
    grid_geopot_half[:,:,k] .= grid_geopot_half[:,:,k+1] .+ rdgas*virtual_t[:,:,k] .* (grid_lnp_half[:,:,k+1] - grid_lnp_half[:,:,k])
  end
  
  for k=1:nd
    #Φ_{k} = Φ_{k+1/2} + RT_k(ln p_{k+1/2} - ln p_{k})
    grid_geopot_full[:,:,k] .= grid_geopot_half[:,:,k+1] .+ rdgas*virtual_t[:,:,k] .* (grid_lnp_half[:,:,k+1] - grid_lnp_full[:,:,k])
  end
  
end 




