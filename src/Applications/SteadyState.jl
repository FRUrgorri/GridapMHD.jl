"""
  SteadyState(; <keyword arguments>)

Driver capable of running 3D cases and fully developed approximation cases with solid
coupling in a rectangular geometry.

# Arguments
- To be defined
"""
function SteadyState(;
  backend = nothing,
  np = nothing,
  title = "MHD_SS",
#  nruns = 1,  #There is no interest a priory to repeate the same computation more than once (other than sudy scalability)
  path = ".",
  kwargs...
)

#  for ir in 1:nruns
#    _title = title*"_r$ir"
    if isa(backend,Nothing)
      @assert isa(np,Nothing)
      info, t, xh, Ω = _SteadyState(;title=title,path=path,kwargs...)
    else
      @assert backend ∈ [:sequential,:mpi]
      @assert !isa(np,Nothing)
      if backend === :sequential
        info, t, xh, Ω = with_debug() do distribute
          _SteadyState(;distribute=distribute,rank_partition=np,title=title,path=path,kwargs...)
        end
      else
        info,t, xh, Ω  = with_mpi() do distribute
          _SteadyState(;distribute=distribute,rank_partition=np,title=title,path=path,kwargs...)
        end
      end
 #   end

    info[:np] = np
    info[:backend] = backend
    info[:title] = title
    map_main(t.data) do data
      for (k,v) in data
        info[Symbol("time_$k")] = v.max
      end
      save(joinpath(path,"$title.bson"),info)
    end
  end

#  nothing
  return xh, Ω
end

function _SteadyState(;
  title = "MHD_SS",
  path = ".",
  distribute = nothing,
  rank_partition = nothing,
#  nl = (4,4),
#  ns = (2,2),
  modelGen = nothing,              
#  domain_tags = ("fluid",),           
  normalization = :mhd,              # :mhd or :cfd   
  Ha = 10.0,
  Re = 1.0,
  N = nothing,
  convection = true,
  Bfield = VectorValue(0.0,1.0,0.0),  #This is B/B_0 where B_0 is the one used to define Ha, it is in general a function of x,y,z
#  curl_free = false,
#  b = 1.0,
#  L = nothing,
#  tw_Ha = 0.0,
#  tw_s = 0.0,
#  cw_Ha = 0.0,
#  cw_s = 0.0, 
  u_inlet = VectorValue(0.0,0.0,1.0), #This is U/U_0 where U_0 is the one used to define Re, it is in general a function of x,y,z
#  vtk = true,
  solve = true,
  solver = :julia,
  verbose = true,
  mesh2vtk = false,
  source = VectorValue(0.0, 0.0, 0.0),
#  fluid_stretching = :Roberts,
#  fluid_stretch_params = (0.5, 1.0),
#  μ = 0.0,
  ku = 2,
  kj = 1,
#  τ_Ha = 100.0,
#  τ_s = 100.0,
#  res_assemble = false,
#  jac_assemble = false,
#  nsums = 10,
  #petsc_options="-snes_monitor -ksp_error_if_not_converged true -ksp_converged_reason -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps"
)

  # The total number of mesh elements is the liquid region elements plus
  # ns elements in the direction of the solid domain.
  # 2*ns accounts for ns solid cells on each side of the liquid for each
  # direction
#  nc = nl .+ (2 .* ns)
  
 
  info = Dict{Symbol,Any}()

  params = Dict{Symbol,Any}(
    :solve=>solve,
#    :res_assemble=>res_assemble,
#    :jac_assemble=>jac_assemble,
  )

  # Communicator
  if isa(distribute,Nothing)
    @assert isa(rank_partition,Nothing)
    rank_partition = Tuple(fill(1,3))     #Always 3D problems (even FD are computationally 3D)
    distribute = DebugArray
  end
#  @assert length(rank_partition) == length(nc)
  parts = distribute(LinearIndices((prod(rank_partition),)))
  
"""
  # Check for FD approx. (2D) or full 3D simulation
  FD = (
    (length(nc) == 2) && (length(rank_partition) == 2) && isa(L, Nothing) &&
    (Re == 1.0) && isa(N, Nothing)
  )
  Full3D = (
    (length(nc) == 3) && (length(rank_partition) == 3) && isa(L, Number) &&
    isa(Re, Nothing) && isa(N, Number)
  )
  if FD
    _nc = (nc[1],nc[2],3)
    _rank_partition = (rank_partition[1], rank_partition[2], 1)
    L = 0.1
    N = Ha^2/Re
    f = VectorValue(0.0, 0.0, 1.0)
    periodic = (false, false, true)
  elseif Full3D
    _nc = nc
    _rank_partition = rank_partition
    Re = Ha^2/N
    f = VectorValue(0.0, 0.0, 0.0)
    periodic = (false, false, false)
    if ns[3] > 0
      error("No solid elements allowed at inlet/outlet regions.")
    end
  end
"""
  # Timer
  t = PTimer(parts,verbose=verbose)
  params[:ptimer] = t
  tic!(t,barrier=true)

  # Solver
  if isa(solver,Symbol)
    solver = default_solver_params(Val(solver))
  end
  params[:solver] = solver
  
  #FE order
  params[:fespaces] = Dict{Symbol, Any}(
  :ku => ku,
  :kj => kj,
  )

  # Domain definitions
#  domain_liq = (-b, b, -1.0, 1.0, 0.0, L)
#  domain = domain_liq .+ (-tw_s, tw_s, -tw_Ha, tw_Ha, 0.0, 0.0)

  # Reduced quantities
  @assert normalization ∈ [:mhd,:cfd]
  if normalization === :mhd
    α = 1.0/N
    β = 1.0/Ha^2
    γ = 1.0
  else
    α = 1.0
    β = 1.0/Re
    γ = N
  end
"""
  # Prepare problem in terms of reduced quantities
  _mesh_map = Meshers.solid_mesh_map(
    Ha,
    b,
    tw_Ha,
    tw_s,
    nc,
    nl,
    ns,
    domain,
    fluid_stretching,
    fluid_stretch_params,
  )
  model = CartesianDiscreteModel(
    parts, _rank_partition, domain, _nc;
    isperiodic=periodic, map=_mesh_map
  )
  Meshers.solid_add_tags!(model, b, tw_Ha, tw_s)
""" 
  #Model from the input  function
  
  @assert !isa(modelGen,Nothing)
  model = modelGen(parts,rank_partition)
  
  params[:model] = model
  Ω = Interior(model)

  if mesh2vtk
    meshpath = joinpath(path, title*"_mesh")
    mkpath(meshpath)
    writevtk(model, meshpath)
  end
  
  
  
  params[:fluid] = Dict{Symbol, Any}(
    :domain=>nothing, #For the moment, only fluid
    :α=>α,
    :β=>β,
    :γ=>γ,
#    :f=>f,
    :f=>source,
    :B=>Bfield,
    :ζ=>0.0,
    :convection=>convection,
  )
   
 """ 
  if (tw_s > 0.0) || (tw_Ha > 0.0)
    params[:fluid][:domain] = "fluid"
    σ_Ω = σ_field(model, Ω, cw_Ha, cw_s, tw_Ha, tw_s)
    params[:solid] = Dict(:domain=>"solid", :σ=>σ_Ω)
  end
"""
  # Boundary conditions
"""
  insulated_tags, thinWall_params, noslip_extra_tags = Meshers.solid_wall_BC(
    model, cw_Ha, cw_s, tw_Ha, tw_s, τ_Ha, τ_s
  )
  noslip_tags = append!(["fluid-solid-boundary"], noslip_extra_tags)
  if FD
    u_BC = Dict(:tags=>noslip_tags)
    j_BC = Dict(:tags=>insulated_tags)
"""
#  elseif Full3D
    u_BC = Dict(
#      :tags=>["inlet", noslip_tags...],
      :tags=>["inlet", "noslip"],  #What happen if there is no inlet
      :values=>[
        u_inlet,
#       fill(VectorValue(0.0, 0.0, 0.0), length(noslip_tags))...
        VectorValue(0.0,0.0,0.0)
      ]
    )
    j_BC = Dict(
#      :tags=>[insulated_tags..., "inlet", "outlet"],
      :tags=>["insulated","inlet","outlet"],
    )

#  end
  params[:bcs] = Dict(
    :u=>u_BC,
    :j=>j_BC,
#    :thin_wall=>thinWall_params, #TBD
  )

""" TBD: Allow a more general stabilization (at least a bit)
  # Stabilization method
  if μ > 0
    ĥ = b/nl[1]    See how the cell size is computed in GridapTritium
    params[:bcs][:stabilization] = Dict(:μ=>μ*ĥ, :domain=>"fluid")
  end
"""
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
 # tic!(t,barrier=true)

  # Post process
"""
  if FD
    cellfields, uh_0, kp = postprocess_FD(xh, Ω, b, cw_s, cw_Ha)
  elseif Full3D
    cellfields, uh_0, kp = postprocess_3D(xh, model, Ω, b)
  end

  if cw_s == 0.0 && cw_Ha == 0.0
    kp_a = kp_shercliff_cartesian(b, Ha)
  else
    kp_a = kp_tillac(b, Ha, cw_s, cw_Ha)
  end
  dev_kp = 100*abs(kp_a - kp)/max(kp_a, kp)
"""
"""

  if vtk
    if (tw_Ha > 0.0) && (tw_s > 0.0)
      push!(cellfields, "σ"=>σ_Ω)
    end
     if !isa (σ,Nothing)
        push!(cellfields, "σ"=>σ)
     end if
    if B_func != :uniform
      push!(cellfields, "B"=>CellField(Bfield, Ω))
    end
    writevtk(Ω, joinpath(path, title), order=max(ku,kj), cellfields=cellfields)
    toc!(t,"vtk")
  end
"""


  if verbose
    display(t)
  end

#  cellfields_dict = Dict(cellfields)


#This info is now much limited
#  info[:nc] = nc
  info[:ncells] = num_cells(model)
  info[:ndofs_u] = length(get_free_dof_values(xh[1]))
  info[:ndofs_p] = length(get_free_dof_values(xh[2]))
  info[:ndofs_j] = length(get_free_dof_values(xh[3]))
  info[:ndofs_φ] = length(get_free_dof_values(xh[4]))
  info[:ndofs] = length(get_free_dof_values(xh))
  info[:Re] = Re
  info[:Ha] = Ha
  info[:N] = N
#  info[:cw] = cw

""" 
  info[:cw_s] = cw_s
  info[:cw_Ha] = cw_Ha
  info[:τ_s] = τ_s
  info[:τ_Ha] = τ_Ha
  info[:b] = b
  info[:L] = L
  info[:uh_0] = uh_0
  info[:kp] = kp
  info[:kp_a] = kp_a
  info[:dev_kp] = dev_kp
"""
  info[:convection] = convection
#  info[:μ] = μ

  return info, t, xh, Ω 
end

