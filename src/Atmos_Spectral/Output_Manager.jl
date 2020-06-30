export Output_Manager, Update_Output!, Finalize_Output!
export Lat_Lon_Pcolormesh, Zonal_Mean, Sigma_Zonal_Mean_Pcolormesh, Sigma_Zonal_Mean_Contourf
                             
mutable struct Output_Manager
    nλ::Int64
    nθ::Int64
    nd::Int64
    n_day::Int64
    
    day_to_sec::Int64
    start_time::Int64
    end_time::Int64
    current_time::Int64
    spinup_day::Int64

    λc::Array{Float64, 1}
    θc::Array{Float64, 1}
    σc::Array{Float64, 1}
  
    # nθ × nd × n_day
    # The average is (start, end], namely it does not include the first snapshot.
    t_daily_zonal_mean::Array{Float64, 3}
    t_eq_daily_zonal_mean::Array{Float64, 3}
    u_daily_zonal_mean::Array{Float64, 3}
    v_daily_zonal_mean::Array{Float64, 3}
    
    ps_daily_mean::Array{Float64, 3}

    n_daily_mean::Array{Float64, 1}
    

    # The average from spinup_day+1 to n_day
    t_zonal_mean::Array{Float64, 2}
    t_eq_zonal_mean::Array{Float64, 2}
    u_zonal_mean::Array{Float64, 2}
    v_zonal_mean::Array{Float64, 2}
    ps_mean::Array{Float64, 2}
    
end

function Output_Manager(mesh::Spectral_Spherical_Mesh, vert_coord::Vert_Coordinate, start_time::Int64, end_time::Int64, spinup_day::Int64)
    nλ = mesh.nλ
    nθ = mesh.nθ
    nd = mesh.nd
    
    day_to_sec = 86400
    current_time = start_time

    λc = mesh.λc
    θc = mesh.θc

    #todo definition of sigma coordinate
    bk = vert_coord.bk
    σc = (bk[2:nd+1] + bk[1:nd])/2.0
  
    # nθ × nd × n_day
    # The average is (start, end], namely it does not include the first snapshot.
    n_day = Int64((end_time - start_time)/day_to_sec)
    t_daily_zonal_mean = zeros(Float64, nθ, nd, n_day)
    t_eq_daily_zonal_mean = zeros(Float64, nθ, nd, n_day)
    u_daily_zonal_mean = zeros(Float64, nθ, nd, n_day)
    v_daily_zonal_mean = zeros(Float64, nθ, nd, n_day)
    ps_daily_mean = zeros(Float64, nλ, nθ, n_day)
    n_daily_mean = zeros(Float64, n_day)

    # The average from spinup_day+1 to n_day
    t_zonal_mean = zeros(Float64, nθ, nd)
    t_eq_zonal_mean = zeros(Float64, nθ, nd)
    u_zonal_mean = zeros(Float64, nθ, nd)
    v_zonal_mean = zeros(Float64, nθ, nd)
    ps_mean = zeros(Float64, nλ, nθ)

    Output_Manager(nλ, nθ, nd, n_day,
    day_to_sec, start_time, end_time, current_time, spinup_day,
    λc, θc, σc,
    t_daily_zonal_mean, t_eq_daily_zonal_mean, u_daily_zonal_mean, v_daily_zonal_mean, 
    ps_daily_mean, n_daily_mean, 
    t_zonal_mean,t_eq_zonal_mean, u_zonal_mean, v_zonal_mean, ps_mean)
end

function Update_Output!(output_manager::Output_Manager, dyn_data::Dyn_Data, current_time::Int64)
    @assert(current_time > output_manager.current_time)
    output_manager.current_time = current_time
    day_to_sec, start_time, n_day = output_manager.day_to_sec, output_manager.start_time, output_manager.n_day

    t_daily_zonal_mean, t_eq_daily_zonal_mean, u_daily_zonal_mean, v_daily_zonal_mean, ps_daily_mean, n_daily_mean = 
    output_manager.t_daily_zonal_mean, output_manager.t_eq_daily_zonal_mean,
    output_manager.u_daily_zonal_mean, output_manager.v_daily_zonal_mean, 
    output_manager.ps_daily_mean, output_manager.n_daily_mean


    i_day = div(current_time - start_time - 1, day_to_sec) + 1

    if(i_day > n_day)
        @info "Warning: i_day > n_day in Output_Manager:Update!"
        return 
    end
    
    t_daily_zonal_mean[:,:,i_day] .+= Zonal_Mean(dyn_data.grid_t_c)
    t_eq_daily_zonal_mean[:,:,i_day] .+= Zonal_Mean(dyn_data.grid_t_eq)
    u_daily_zonal_mean[:,:,i_day] .+= Zonal_Mean(dyn_data.grid_u_c)
    v_daily_zonal_mean[:,:,i_day] .+= Zonal_Mean(dyn_data.grid_v_c)

    ps_daily_mean[:,:,i_day] .+= dyn_data.grid_ps_c[:,:,1]

    n_daily_mean[i_day] += 1
end

function Finalize_Output!(output_manager::Output_Manager, save_file_name::String = "None", mean_save_file_name::String = "None")

    n_day = output_manager.n_day

    t_daily_zonal_mean, t_eq_daily_zonal_mean, u_daily_zonal_mean, v_daily_zonal_mean, ps_daily_mean, n_daily_mean = 
    output_manager.t_daily_zonal_mean, output_manager.t_eq_daily_zonal_mean,
    output_manager.u_daily_zonal_mean, output_manager.v_daily_zonal_mean, 
    output_manager.ps_daily_mean, output_manager.n_daily_mean
    
    for i_day = 1:n_day
        t_daily_zonal_mean[:,:,i_day] ./= n_daily_mean[i_day]
        t_eq_daily_zonal_mean[:,:,i_day] ./= n_daily_mean[i_day]
        u_daily_zonal_mean[:,:,i_day] ./= n_daily_mean[i_day]
        v_daily_zonal_mean[:,:,i_day] ./= n_daily_mean[i_day]
        ps_daily_mean[:,:,i_day] ./= n_daily_mean[i_day]
        n_daily_mean[i_day] = 1.0
    end

    spinup_day = output_manager.spinup_day
    t_zonal_mean, t_eq_zonal_mean, u_zonal_mean, v_zonal_mean, ps_mean = 
    output_manager.t_zonal_mean, output_manager.t_eq_zonal_mean, 
    output_manager.u_zonal_mean, output_manager.v_zonal_mean, output_manager.ps_mean

    t_zonal_mean .= dropdims(mean(t_daily_zonal_mean[:,:,spinup_day+1:n_day], dims=3), dims=3)
    t_eq_zonal_mean .= dropdims(mean(t_eq_daily_zonal_mean[:,:,spinup_day+1:n_day], dims=3), dims=3)
    u_zonal_mean .= dropdims(mean(u_daily_zonal_mean[:,:,spinup_day+1:n_day], dims=3), dims=3)
    v_zonal_mean .= dropdims(mean(v_daily_zonal_mean[:,:,spinup_day+1:n_day], dims=3), dims=3)
    ps_mean .= dropdims(mean(ps_daily_mean[:,:,spinup_day+1:n_day], dims=3), dims=3)
       
    if save_file_name != "None"
        @save save_file_name output_manager
    end

    if mean_save_file_name != "None"
        @save mean_save_file_name t_zonal_mean u_zonal_mean v_zonal_mean
    end
end





function Sigma_Zonal_Mean_Contourf(output_manager::Output_Manager, save_file_pref::String)
    
    θc = output_manager.θc
    σc = output_manager.σc
    nθ = length(θc)
    θc_deg = θc*180/pi
    nd = output_manager.nd
    
    X,Y = repeat(θc_deg, 1, nd), repeat(σc, 1, nθ)'

    t_zonal_mean, t_eq_zonal_mean, u_zonal_mean, v_zonal_mean = output_manager.t_zonal_mean, output_manager.t_eq_zonal_mean, output_manager.u_zonal_mean, output_manager.v_zonal_mean
   
    PyPlot.contourf(X, Y, t_zonal_mean, levels = 10)
    PyPlot.gca().invert_yaxis()
    PyPlot.xlabel("Latitude")
    PyPlot.ylabel("σ")
    PyPlot.colorbar()
    PyPlot.savefig(save_file_pref * "_T.png")
    PyPlot.close("all")

    PyPlot.contourf(X, Y, t_eq_zonal_mean, levels = 10)
    PyPlot.gca().invert_yaxis()
    PyPlot.xlabel("Latitude")
    PyPlot.ylabel("σ")
    PyPlot.colorbar()
    PyPlot.savefig(save_file_pref * "_Teq.png")
    PyPlot.close("all")

    PyPlot.contourf(X, Y, u_zonal_mean, levels = 10)
    PyPlot.gca().invert_yaxis()
    PyPlot.xlabel("Latitude")
    PyPlot.ylabel("σ")
    PyPlot.colorbar()
    PyPlot.savefig(save_file_pref * "_U.png")
    PyPlot.close("all")

    PyPlot.contourf(X, Y, v_zonal_mean, levels = 10)
    PyPlot.gca().invert_yaxis()
    PyPlot.xlabel("Latitude")
    PyPlot.ylabel("σ")
    PyPlot.colorbar()
    PyPlot.savefig(save_file_pref * "_V.png")
    PyPlot.close("all")
    
    
end


function Sigma_Zonal_Mean_Pcolormesh(output_manager::Output_Manager, save_file_pref::String)
    
    θc = output_manager.θc
    σc = output_manager.σc
    nθ = length(θc)
    θc_deg = θc*180/pi
    nd = output_manager.nd
    
    X,Y = repeat(θc_deg, 1, nd), repeat(σc, 1, nθ)'

    t_zonal_mean, u_zonal_mean, v_zonal_mean = output_manager.t_zonal_mean, output_manager.u_zonal_mean, output_manager.v_zonal_mean
   
    PyPlot.pcolormesh(X, Y, t_zonal_mean, shading= "gouraud", cmap="viridis")
    PyPlot.gca().invert_yaxis()
    PyPlot.colorbar()
    PyPlot.savefig(save_file_pref * "_T.png")
    PyPlot.close("all")

    PyPlot.pcolormesh(X, Y, u_zonal_mean, shading= "gouraud", cmap="viridis")
    PyPlot.gca().invert_yaxis()
    PyPlot.colorbar()
    PyPlot.savefig(save_file_pref * "_U.png")
    PyPlot.close("all")

    PyPlot.pcolormesh(X, Y, v_zonal_mean, shading= "gouraud", cmap="viridis")
    PyPlot.gca().invert_yaxis()
    PyPlot.colorbar()
    PyPlot.savefig(save_file_pref * "_V.png")
    PyPlot.close("all")
    
    
end


function Lat_Lon_Pcolormesh(output_manager::Output_Manager, grid_dat::Array{Float64,3}, level::Int64, save_file_name::String = "None")
    
    λc, θc = output_manager.λc, output_manager.θc
    nλ, nθ = length(λc), length(θc)
    λc_deg, θc_deg = λc*180/pi, θc*180/pi
    
    
    X,Y = repeat(λc_deg, 1, nθ), repeat(θc_deg, 1, nλ)'
    
    
    PyPlot.pcolormesh(X, Y, grid_dat[:,:,level], shading= "gouraud", cmap="viridis")
    PyPlot.axis("equal")
    PyPlot.colorbar()
    
    if save_file_name != "None"
        PyPlot.savefig(save_file_name)
        PyPlot.close("all")
    end
    
end


function Lat_Lon_Pcolormesh(mesh::Spectral_Spherical_Mesh, grid_dat::Array{Float64,3}, level::Int64, save_file_name::String = "None")
    
    λc, θc = mesh.λc, mesh.θc
    nλ, nθ = length(λc), length(θc)
    λc_deg, θc_deg = λc*180/pi, θc*180/pi
    
    X,Y = repeat(λc_deg, 1, nθ), repeat(θc_deg, 1, nλ)'
    
    
    PyPlot.pcolormesh(X, Y, grid_dat[:,:,level], shading= "gouraud", cmap="viridis")
    PyPlot.axis("equal")
    PyPlot.colorbar()
    
    if save_file_name != "None"
        PyPlot.savefig(save_file_name)
        PyPlot.close("all")
    end
    
end


function Zonal_Mean(grid_dat::Array{Float64,3})
    
    return dropdims(mean(grid_dat, dims=1), dims=1)
    
end


function Sigma_Zonal_Mean_Pcolormesh(output_manager::Output_Manager,
    zonal_mean_data::Array{Float64,2}, save_file_name::String = "None")
    
    θc = output_manager.θc
    σc = output_manager.σc
    nθ = length(θc)
    θc_deg = θc*180/pi
    nd = output_manager.nd
    
    X,Y = repeat(θc_deg, 1, nd), repeat(σc, 1, nθ)'
    
    
    PyPlot.pcolormesh(X, Y, zonal_mean_data, shading= "gouraud", cmap="viridis")
    PyPlot.colorbar()
    PyPlot.gca().invert_yaxis()
    
    if save_file_name != "None"
        PyPlot.savefig(save_file_name)
        PyPlot.close("all")
    end
    
end


function Sigma_Zonal_Mean_Contourf(output_manager::Output_Manager, 
    zonal_mean_data::Array{Float64,2}, save_file_name::String = "None")
    
    θc = output_manager.θc
    σc = output_manager.σc
    nθ = length(θc)
    θc_deg = θc*180/pi
    nd = output_manager.nd
    
    X,Y = repeat(θc_deg, 1, nd), repeat(σc, 1, nθ)'
    
    PyPlot.contourf(X, Y, zonal_mean_data, levels = 10)
    PyPlot.gca().invert_yaxis()
    
    if save_file_name != "None"
        PyPlot.savefig(save_file_name)
        PyPlot.close("all")
    end
    
end