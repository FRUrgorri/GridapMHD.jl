function SolidFD(;
  backend = nothing,
  np = nothing,
  parts = nothing,
  title = "SolidFD",
  nruns = 1,
  path = ".",
  kwargs...)

  for ir in 1:nruns
    _title = title*"_r$ir"
    if isa(backend,Nothing)
      @assert isa(np,Nothing)
      info, t = _SolidFD(;title=_title,path=path,kwargs...)
    else
      @assert backend ∈ [:sequential,:mpi]
      @assert !isa(np,Nothing)
      if backend === :sequential
        info, t = with_debug() do distribute
          _SolidFD(;distribute=distribute,rank_partition=np,title=_title,path=path,kwargs...)
        end
      else
        info,t = with_mpi() do distribute
          _SolidFD(;distribute=distribute,rank_partition=np,title=_title,path=path,kwargs...)
        end
      end
    end

    info[:np] = np
    info[:backend] = backend
    info[:title] = title
    map_main(t.data) do data
      for (k,v) in data
        info[Symbol("time_$k")] = v.max
      end
      save(joinpath(path,"$_title.bson"),info)
    end
  end

  nothing
end

function _SolidFD(;
  distribute = nothing,
  rank_partition = nothing,
  nl = (4,4),
  ns = (2,2),
  Ha = 10.0,
  dir_B = (0.0,1.0,0.0),
  cw_Ha = 0.0,
  cw_s = 0.0,
  b = 1.0,
  tw_Ha = 0.0,
  tw_s = 0.0,
  title = "SolidFD",
  path = ".",
  vtk = true,
  debug = false,
  res_assemble = false,
  jac_assemble = false,
  solve = true,
  solver = :julia,
  verbose = true,
  mesh = false,
  stretch_fine = false,
  τ_Ha = 100.0,
  τ_s = 100.0,
  nsums = 10,
  #petsc_options="-snes_monitor -ksp_error_if_not_converged true -ksp_converged_reason -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps"
  )

  # The total number of mesh elements is the liquid region elements plus
  # ns elements in the direction of the solid domain.
  # 2*ns accounts for ns solid cells on each side of the liquid for each
  # direction
  nc = nl .+ 2*ns

  if ((cw_Ha == 0.0) & (tw_Ha > 0.0)) | ((cw_s == 0.0) & (tw_s > 0.0))
    error("Insulator wall BC (cw = 0) requires thin wall approximation (tw = 0)")
  end

  info = Dict{Symbol,Any}()

  params = Dict{Symbol,Any}(
    :debug=>debug,
    :solve=>solve,
    :res_assemble=>res_assemble,
    :jac_assemble=>jac_assemble,
  )

  # Communicator
  if isa(distribute,Nothing)
    @assert isa(rank_partition,Nothing)
    rank_partition = Tuple(fill(1,length(nc)))
    distribute = DebugArray
  end
  @assert length(rank_partition) == length(nc)
  parts = distribute(LinearIndices((prod(rank_partition),)))
  
  t = PTimer(parts,verbose=verbose)
  params[:ptimer] = t
  tic!(t,barrier=true)

  if isa(solver,Symbol)
    solver = default_solver_params(Val(solver))
  end
  params[:solver] = solver

  domain_liq = (-b,b,-1.0,1.0,0.0,0.1)
  domain = domain_liq .+ (-tw_s, tw_s, -tw_Ha, tw_Ha, 0.0, 0.0)

  # Reduced quantities
  Re = 1.0
  N = Ha^2/Re
  dirB = (1/norm(VectorValue(dir_B)))*VectorValue(dir_B)
  α = 1.0/N
  β = 1.0/Ha^2
  γ = 1.0

  # Prepare problem in terms of reduced quantities
  partition = (nc[1],nc[2],3)
  ranks = (rank_partition[1],rank_partition[2],1)
  model = CartesianDiscreteModel(
    parts,ranks,domain,partition;isperiodic=(false,false,true),map=meshmap)
  Ω = Interior(model)
  params[:model] = model

  labels = get_face_labeling(model)
  tags_j_Ha = append!(collect(1:20),[23,24])
  tags_j_side = append!(collect(1:20),[25,26])
  tags_outlet = append!(collect(1:20),[22])

  add_tag_from_tags!(labels,"Ha_walls",tags_j_Ha)
  add_tag_from_tags!(labels,"side_walls",tags_j_side)
  add_tag_from_tags!(labels,"outlet",tags_outlet)
  
  if mesh
    writevtk(model,"Mesh")
  end

  params[:fluid] = Dict(
    :domain=>nothing,
    :α=>α,
    :β=>β,
    :γ=>γ,
    :f=>VectorValue(0.0,0.0,1.0),
    :B=>dirB,
    :ζ=>0.0,
  )

  params[:bcs] =  Dict(
    :u=>Dict(:tags=>["Ha_walls","side_walls"]),
    :j=>Dict{Symbol,Vector{String}}(),
    :thin_wall=>Vector{Dict{Symbol,Any}}() 
  )
  

  insulated_tags = Vector{String}()
  thinWall_options = Vector{Dict{Symbol,Any}}() 

  if cw_Ha == 0.0 
    push!(insulated_tags,"Ha_walls")
  else
    push!(thinWall_options, Dict(
        :cw => cw_Ha,
        :τ => τ_Ha, 
        :domain =>Boundary(model, tags="Ha_walls")
        )
      ) 
  end  

  if cw_s == 0.0
    push!(insulated_tags,"side_walls")
  else
    push!(thinWall_options, Dict(
        :cw => cw_s,
        :τ => τ_s,
        :domain =>Boundary(model, tags="side_walls")
        )
      )
  end
 
  params[:bcs][:j] = Dict(:tags=>insulated_tags)

  if thinWall_options != []
    params[:bcs][:thin_wall] = thinWall_options
  end
 
  toc!(t,"pre_process")

  # Solve it
  if !uses_petsc(params[:solver])
    xh,fullparams,info = main(params;output=info)
  else
    petsc_options = params[:solver][:petsc_options]
    xh,fullparams,info = GridapPETSc.with(args=split(petsc_options)) do
      xh,fullparams,info = main(params;output=info)
      GridapPETSc.gridap_petsc_gc() # Destroy all PETSc objects
      return xh,fullparams,info
    end
  end
  
  t = fullparams[:ptimer]

  # Compute quantities

  tic!(t,barrier=true)
  uh,ph,jh,φh = xh
  
  div_jh = ∇·jh 
  div_uh = ∇·uh  
  
  
  #Post process
  
  dΩ = Measure(Ω,6)
  uh_0 = sum(∫(uh)*dΩ)[3]/sum(∫(1.0)*dΩ)
 
  uh_n = uh/uh_0
  ph_n = ph/uh_0
  jh_n = jh/uh_0
  φh_n = φh/uh_0  

  div_jh_n = div_jh/uh_0
  div_uh_n = div_uh/uh_0

  if cw_s == 0.0
    u_a(x) = analytical_GeneralHunt_u(1.0,cw_Ha,-1.0,Ha,nsums,x)
    e_u = u_a - uh_n
  else
    e_u = uh_n
  end

  kp = 1/uh_0

  if cw_s == 0.0 && cw_Ha == 0.0
    kp_a = kp_shercliff_cartesian(b,Ha)
  else
    kp_a = kp_tillac(b,Ha,cw_s,cw_Ha)
  end

  dev_kp = 100*abs(kp_a-kp)/max(kp_a,kp)

  if vtk
    writevtk(Ω,joinpath(path,title),
      order=2,
      cellfields=[
        "uh"=>uh_n,"e_u"=>e_u,"ph"=>ph_n,"jh"=>jh_n,"phi"=>φh_n,"div_jh"=>div_jh_n,"div_uh"=>div_uh_n])
    toc!(t,"vtk")
  end
  if verbose
    display(t)
  end

  info[:nc] = nc
  info[:ncells] = num_cells(model)
  info[:ndofs_u] = length(get_free_dof_values(uh))
  info[:ndofs_p] = length(get_free_dof_values(ph))
  info[:ndofs_j] = length(get_free_dof_values(jh))
  info[:ndofs_φ] = length(get_free_dof_values(φh))
  info[:ndofs] = length(get_free_dof_values(xh))
  info[:Re] = Re
  info[:Ha] = Ha
  info[:N] = N
  info[:cw_s] = cw_s
  info[:τ_s] = τ_s
  info[:cw_Ha] = cw_Ha
  info[:τ_Ha] = τ_Ha
  info[:b] = b
  info[:uh_0] = uh_0
  info[:kp] = kp
  info[:kp_a] =kp_a
  info[:dev_kp] = dev_kp 
  info, t
end

# Mesher maps and helper funcs
"""
  mesh_map(coord)

Map function to pass to CartesianDiscreteModel.

Combines the solid mesh map and the liquid stretchMHD map as required depending
on the input tw_s, tw_Ha values and the stretch_fine toggle.
"""
function mesh_map(coord)
  stretch_Ha = sqrt(Ha/(Ha-1))
  stretch_side = sqrt(sqrt(Ha)/(sqrt(Ha)-1))

  if (tw_s > 0.0) | (tw_Ha > 0.0)
    ncoord = solidMap(ncoord, tw=(tw_s, tw_Ha), nc=nc, ns=ns, nl=nl, domain=domain)
  end

  if stretch_fine
    ncoord = stretchMHD(coord,domain=(0,-b,0,-1.0),factor=(stretch_Ha,stretch_Ha),dirs=(1,2))
    ncoord = stretchMHD(ncoord,domain=(0,b,0,1.0),factor=(stretch_Ha,stretch_Ha),dirs=(1,2))
  else
    ncoord = stretchMHD(coord,domain=(0,-b,0,-1.0),factor=(stretch_side,stretch_Ha),dirs=(1,2))
    ncoord = stretchMHD(ncoord,domain=(0,b,0,1.0),factor=(stretch_side,stretch_Ha),dirs=(1,2))
  end
  ncoord
end

"""
  solidMap(coord, tw, nc, ns, nl, domain)

Map to space out the mesh nodes evenly in the solid region according to the
specified number of solid cells, ns, and the solid width, tw.  It also requires
the total number of cells, nc, the number of cells in the liquid region, nl, and
the domain.

tw, nc, ns, and nl are assumed to be arrays corresponding to the X and Y
directions respectevely.

domain is an array specifying the domain limits:
  domain = (x0, xf, y0, yf,...)
"""
function solidMap(coord, tw, nc, ns, nl, domain)
  ncoord = collect(coord.data)
  x0 = domain[1]
  xf = domain[2]
  y0 = domain[3]
  yf = domain[4]
  dx = (xf - x0)/nc[1]
  dxl = (xf - x0 - 2*tw[1])/nl[1]
  dxs = tw[1]/ns[1]
  dy = (yf - y0)/nc[2]
  dyl = (yf - y0 - 2*tw[2])/nl[2]
  dys = tw[2]/ns[2]

  if tw_s > 0.0
    nx = abs(ncoord[1] - x0)/dx
    if nx < ns[1]
      ncoord[1] = x0 + nx*dxs
    elseif ns[1] <= nx <= (nl[1] + ns[1])
      ncoord[1] = x0 + tw[1] + (nx - ns[1])*dxl
    elseif nx > (nl[1] + ns[1])
      ncoord[1] = x0 + tw[1] + nl[1]*dxl + (nx - nl[1] - ns[1])*dxs
    end
  elseif tw_Ha > 0.0
    ny = abs(ncoord[2] - y0)/dy
    if ny < ns[2]
      ncoord[2] = y0 + ny*dys
    elseif ns[2] <= ny <= (nl[2] + ns[2])
      ncoord[2] = y0 + tw[2] + (ny - ns[2])*dyl
    elseif ny > (nl[2] + ns[2])
      ncoord[2] = y0 + tw[2] + nl[2]*dyl + (ny - nl[2] - ns[2])*dys
    end
  end
  VectorValue(ncoord)
end

function stretchMHD(coord;domain=(0.0,1.0,0.0,1.0,0.0,1.0),factor=(1.0,1.0,1.0),dirs=(1,2,3))
  ncoord = collect(coord.data)
  for (i,dir) in enumerate(dirs)
    ξ0 = domain[i*2-1]
    ξ1 = domain[i*2]
    l =  ξ1 - ξ0
    c = (factor[i] + 1)/(factor[i] - 1)

    if l > 0
      if ξ0 <= coord[dir] <= ξ1
        ξx = (coord[dir] - ξ0)/l                     # ξx from 0 to 1 uniformly distributed
        ξx_streched = factor[i]*(c^ξx-1)/(1+c^ξx)    # ξx streched from 0 to 1 towards 1
        ncoord[dir] =  ξx_streched*l + ξ0            # coords streched towards ξ1
      end
    else
      if ξ1 <= coord[dir] <= ξ0
        ξx = (coord[dir] - ξ0)/l                     # ξx from 0 to 1 uniformly distributed
        ξx_streched = factor[i]*(c^ξx-1)/(1+c^ξx)    # ξx streched from 0 to 1 towards 1
        ncoord[dir] =  ξx_streched*l + ξ0            # coords streched towards ξ1
      end
    end
  end
  return VectorValue(ncoord)
end

# Analytical solution
function analytical_GeneralHunt_u(
  #General Hunt analytical formula (d_b = 0 for Shercliff)
  l::Float64,       # channel aspect ratio
  d_b::Float64,     # Hartmann walls conductivity ratio
  grad_pz::Float64, # Dimensionless (MHD version) presure gradient
  Ha::Float64,      # Hartmann number
  n::Int,           # number of sumands included in Fourier series
  x)                # evaluation point normaliced by the Hartmann characteristic lenght

  V = 0.0; V0=0.0;
  for k in 0:n
    α_k = (k + 0.5)*π/l
    N = (Ha^2 + 4*α_k^2)^(0.5)
    r1_k = 0.5*( Ha + N)
    r2_k = 0.5*(-Ha + N)
    
    eplus_1k = 1 + exp(-2*r1_k)
    eminus_1k = 1 - exp(-2*r1_k)
    eplus_2k = 1 + exp(-2*r2_k)
    eminus_2k = 1 - exp(-2*r2_k)
    eplus_k = 1 + exp(-2*(r1_k+r2_k))
    e_x_1k = 0.5*(exp(-r1_k*(1-x[2]))+exp(-r1_k*(1+x[2])))
    e_x_2k = 0.5*(exp(-r2_k*(1-x[2]))+exp(-r2_k*(1+x[2])))
    
    V2 = ((d_b*r2_k + eminus_2k/eplus_2k)*e_x_1k)/(0.5*N*d_b*eplus_1k + eplus_k/eplus_2k)
    V3 = ((d_b*r1_k + eminus_1k/eplus_1k)*e_x_2k)/(0.5*N*d_b*eplus_2k + eplus_k/eplus_1k)

    V += 2*(-1)^k*cos(α_k * x[1])/(l*α_k^3) * (1-V2-V3)
  end
  u_z = V*Ha^2*(-grad_pz) 

  VectorValue(0.0*u_z,0.0*u_z,u_z)
end
