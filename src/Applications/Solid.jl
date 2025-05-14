"""
  Solid(; <keyword arguments>)

Driver capable of running 3D cases and fully developed approximation cases with solid
coupling in a rectangular geometry.

# Arguments
- `backend = nothing`: backend for parallelization. Values: nothing, :sequential or :mpi.
- `np = nothing: array describing mesh partitioning for parallelization.
- `title = "Solid": job title used for saved files.
- `nruns = 1: number of runs.
- `path = ".": path where saved files are stored.
- `nl = (4,4)`: array with the number of nodes in the liquid region for each direction.
- `ns = (2,2)`: array with the number of nodes in the solid region for each direction.
- `Ha = 10.0`: Hartmann number.
- `Re = 1.0`: Reynolds number.
- `N = nothing`: interaction parameter.
- `convection = true`: toggle for the weak form convective term.
- `Bfield = (x...) -> VectorValue(0.0, 1.0, 0.0)`: external magnetic field function `B/B₀`
  where `B₀` is assumed to be the one used to define `Ha`. It is a function of `x,y,z`.
- `b = 1.0`: half-width in the direction perpendicular to the external magnetic field.
- `L = nothing`: length in the axial direction.
- `tw_Ha = 0.0`: width of the solid wall in the external magnetic field direction.
- `tw_s = 0.0`: width of the solid wall normal to the external magnetic field.
- `cw_Ha = 0.0`: wall parameter in the external magnetic field direction.
- `cw_s = 0.0`: wall parameter normal to the external magnetic field.
- `u_inlet = (x...) -> VectorValue(0.0, 0.0, 1.0)`: This is `U/U₀` at the inlet (boundary
  condition), where `U₀` is the one used to define `Re`.
  It is in general a function of `x,y,z`.
- `vtk = true`: toggle to save the final results in vtk format.
- `solve = true`: toggle to run the solver.
- `solver = :julia`: solver to be used and additional solver parameters.
- `verbose = true`: print time statistics.
- `mesh2vtk = false`: save the generated model in vtk format.
- `fluid_stretching = :Roberts`: stretching rule for the fluid mesh.
- `fluid_stretch_params = (0.5, 1.0)`: parameters for the fluid mesh stretching.
- `μ = 0.0`: stabilization method coefficient.  Defaults to no stabilization.
- `τ_Ha = 100.0`: penalty term for the thin wall boundary condition in the Ha boundary.
- `τ_s = 100.0`: penalty term for the thin wall boundary condition in the Side boundary.
- `res_assemble = false`: toggle to time the computation of the residual independently.
- `jac_assemble = false`: toggle to time the computation of the jacobian independently.
- `nsums = 10`: number of terms used for the analytical solution computation.

# Fully developed approximation
For the fully developed approximation, the following arguments need be set:
- `nc`, and `np` must be 2-dimensional arrays.
- `L`, and `N` must be set to `nothing`.
- `Re` must be set to `1.0`.
- `u_inlet` is ignored.

# 3D simulation
If a full 3D simulation is to be run, the following arguments need be set:
- `nc`, and `np` must be 3-dimensional arrays.
- `L`, and `N` must be numbers.
- `Re` must be set to `nothing`.

# Mesh stretching
There are two available rules for liquid mesh stretching:
- `:Roberts`: mesh refinement in the boundary layer region.
- `:hyperbolic`: tanh rule for separation of mesh elements.
"""
function Solid(;
  backend = nothing,
  np = nothing,
  title = "Solid",
  nruns = 1,
  path = ".",
  kwargs...
)

  for ir in 1:nruns
    _title = title*"_r$ir"
    if isa(backend,Nothing)
      @assert isa(np,Nothing)
      info, t = _Solid(;title=_title,path=path,kwargs...)
    else
      @assert backend ∈ [:sequential,:mpi]
      @assert !isa(np,Nothing)
      if backend === :sequential
        info, t = with_debug() do distribute
          _Solid(;distribute=distribute,rank_partition=np,title=_title,path=path,kwargs...)
        end
      else
        info,t = with_mpi() do distribute
          _Solid(;distribute=distribute,rank_partition=np,title=_title,path=path,kwargs...)
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

function _Solid(;
  title = "Solid",
  path = ".",
  distribute = nothing,
  rank_partition = nothing,
  nl = (4,4),
  ns = (2,2),
  Ha = 10.0,
  Re = 1.0,
  N = nothing,
  convection = true,
  Bfield = x -> VectorValue(0.0, 1.0, 0.0),
  b = 1.0,
  L = nothing,
  tw_Ha = 0.0,
  tw_s = 0.0,
  cw_Ha = 0.0,
  cw_s = 0.0,
  u_inlet = (x...) -> VectorValue(0.0, 0.0, 1.0),
  vtk = true,
  solve = true,
  solver = :julia,
  verbose = true,
  mesh2vtk = false,
  fluid_stretching = :Roberts,
  fluid_stretch_params = (0.5, 1.0),
  μ = 0.0,
  τ_Ha = 100.0,
  τ_s = 100.0,
  res_assemble = false,
  jac_assemble = false,
  nsums = 10,
  #petsc_options="-snes_monitor -ksp_error_if_not_converged true -ksp_converged_reason -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps"
)

  # The total number of mesh elements is the liquid region elements plus
  # ns elements in the direction of the solid domain.
  # 2*ns accounts for ns solid cells on each side of the liquid for each
  # direction
  nc = nl .+ (2 .* ns)

  info = Dict{Symbol,Any}()
  params = Dict{Symbol,Any}(
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

  # Timer
  t = PTimer(parts,verbose=verbose)
  params[:ptimer] = t
  tic!(t,barrier=true)

  # Solver
  if isa(solver,Symbol)
    solver = default_solver_params(Val(solver))
  end
  params[:solver] = solver

  # Domain definitions
  domain_liq = (-b, b, -1.0, 1.0, 0.0, L)
  domain = domain_liq .+ (-tw_s, tw_s, -tw_Ha, tw_Ha, 0.0, 0.0)

  # Reduced quantities
  α = 1.0/N
  β = 1.0/Ha^2
  γ = 1.0

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
  Ω = Interior(model)

  if mesh2vtk
    meshpath = joinpath(path, title*"_mesh")
    mkpath(meshpath)
    writevtk(model, meshpath)
  end

  params[:model] = model
  params[:fluid] = Dict{Symbol, Any}(
    :domain=>nothing,
    :α=>α,
    :β=>β,
    :γ=>γ,
    :f=>f,
    :B=>Bfield,
    :ζ=>0.0,
    :convection=>convection,
  )
  if (tw_s > 0.0) || (tw_Ha > 0.0)
    params[:fluid][:domain] = "fluid"
    σ_Ω = σ_field(model, Ω, cw_Ha, cw_s, tw_Ha, tw_s)
    params[:solid] = Dict(:domain=>"solid", :σ=>σ_Ω)
  end

  # Boundary conditions
  insulated_tags, thinWall_params, noslip_extra_tags = Meshers.solid_wall_BC(
    model, cw_Ha, cw_s, tw_Ha, tw_s, τ_Ha, τ_s
  )
  noslip_tags = append!(["fluid-solid-boundary"], noslip_extra_tags)
  if FD
    u_BC = Dict(:tags=>noslip_tags)
    j_BC = Dict(:tags=>insulated_tags)
  elseif Full3D
    u_BC = Dict(
      :tags=>["inlet", noslip_tags...],
      :values=>[
        u_inlet,
        fill(VectorValue(0.0, 0.0, 0.0), length(noslip_tags))...
      ],
    )
    j_BC = Dict(
      :tags=>[insulated_tags..., "inlet", "outlet"],
    )
  end
  params[:bcs] = Dict(
    :u=>u_BC,
    :j=>j_BC,
    :thin_wall=>thinWall_params,
  )

  # Stabilization method
  if μ > 0
    ĥ = b/nl[1]
    params[:bcs][:stabilization] = Dict(:μ=>μ*ĥ, :domain=>"fluid")
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

  # Post process

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

  Δp = (-1)*(
    surf_avg(model, xh[2], "outlet"; restrict=isfluid(b)) -
    surf_avg(model, xh[2], "inlet"; restrict=isfluid(b))
  )/L
  # Avg(B²) assumes axially varying fields
  avg_Bsq = quad(x->norm(Bfield(VectorValue([0.0, 0.0, x])))^2, 0.0, L; n=500)/L
  avg_Ha = sqrt(avg_Bsq)*Ha
  Δp_Miyazaki = kp_tillac(b, avg_Ha, cw_s, cw_Ha)*avg_Bsq
  dev_Δp = 100*abs(Δp - Δp_Miyazaki)/max(Δp, Δp_Miyazaki)

  if vtk
    if (tw_Ha > 0.0) && (tw_s > 0.0)
      push!(cellfields, "σ"=>σ_Ω)
    end
    push!(cellfields, "B"=>CellField(Bfield, Ω))
    writevtk(Ω, joinpath(path, title), order=2, cellfields=cellfields)
    toc!(t,"vtk")
  end
  if verbose
    display(t)
  end

  cellfields_dict = Dict(cellfields)

  info[:nc] = nc
  info[:ncells] = num_cells(model)
  info[:ndofs_u] = length(get_free_dof_values(xh[1]))
  info[:ndofs_p] = length(get_free_dof_values(xh[2]))
  info[:ndofs_j] = length(get_free_dof_values(xh[3]))
  info[:ndofs_φ] = length(get_free_dof_values(xh[4]))
  info[:ndofs] = length(get_free_dof_values(xh))
  info[:Re] = Re
  info[:Ha] = Ha
  info[:N] = N
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
  info[:convection] = convection
  info[:μ] = μ
  info[:Δp] = Δp
  info[:Δp_Miyazaki] = Δp_Miyazaki
  info[:dev_Δp] = dev_Δp

  return info, t
end


# Helper functions
"""
  σ_field(model, Ω, cw_Ha::Real, cw_s::Real, tw_Ha::Real, tw_s::Real)

Returns a DistributedCellField with the adequate conductivity for each domain
region.
"""
function σ_field(model, Ω, cw_Ha::Real, cw_s::Real, tw_Ha::Real, tw_s::Real)
  # Conductivity: cw := (σ_w/σ_l)*(tw/a) => (normalized) => σ_w = cw/tw
  σ_l = 1.0
  if tw_Ha > 0.0
    σ_Ha = cw_Ha/tw_Ha
  else
    σ_Ha = nothing
  end
  if tw_s > 0.0
    σ_s = cw_s/tw_s
  else
    σ_s = nothing
  end

  function _σ_field(σ_l, σ_Ha, σ_s, labels, Ω, cell_entity)
    if tw_Ha > 0.0
      solid_Ha = get_tag_entities(labels, "solid_Ha")[1]
    else
      solid_Ha = nothing
    end
    if tw_s > 0.0
      solid_s = get_tag_entities(labels, "solid_s")[1]
    else
      solid_s = nothing
    end

    function entity_to_σ(entity)
      if entity == solid_Ha
        σ_Ha
      elseif entity == solid_s
        σ_s
      else
        σ_l
      end
    end

    σ_CellField = CellField(map(entity_to_σ, cell_entity), Ω)
    return σ_CellField
  end

  cells = get_cell_gids(model)
  labels = get_face_labeling(model)
  σ_f = map(
    local_views(labels), local_views(cells), local_views(Ω)
  ) do labels, partition, trian
    cell_entity = labels.d_to_dface_to_entity[end]
    _σ_field(σ_l, σ_Ha, σ_s, labels, trian, cell_entity[partition.own_to_local])
  end
  return GridapDistributed.DistributedCellField(σ_f, Ω)
end

"""
  isfluid(b)

Returns a step-function which depends on a three-component vector x.  Such function
returns 1.0 if the coordinates given by the vector x fall within the fluid region
and 0.0 otherwise.  The fluid region is defined by -1 < x[1] < 1, -b < x[2] < b.

It is intended as a step function to limit integration or other operations only
to the fluid domain.  This approach avoids creating a 'model_fluid' using tag
'fluid', as suggested possible by Gridap Tutorial, which seems broken for
GridapDistributed models.  However, that approach would be preferred.
"""
function isfluid(b)
  function _isfluid((x, y, z))
    if (-1 <= x <= 1) && (-b <= y <= b)
      return 1.0
    else
      return 0.0
    end
  end

  return _isfluid
end

# Post-processing and analytical solutions
"""
  postprocess_FD(xh, Ω)

Post process operations and computations to be run after a fully developed
approximation solution `xh` is obtained.  `Ω` is the model's interior.
"""
function postprocess_FD(xh, Ω, b, cw_s, cw_Ha)
  uh, ph, jh, φh = xh
  div_jh = ∇·jh
  div_uh = ∇·uh

  _isfluid = isfluid(b)
  dΩ = Measure(Ω, 6)
  uh_0 = sum(∫(uh*_isfluid)*dΩ)[3]/sum(∫(_isfluid)*dΩ)
  kp = 1/uh_0

  uh_n = uh/uh_0
  ph_n = ph/uh_0
  jh_n = jh/uh_0
  φh_n = φh/uh_0

  div_jh_n = div_jh/uh_0
  div_uh_n = div_uh/uh_0

  if cw_s == 0.0
    u_a(x) = analytical_GeneralHunt_u(1.0, cw_Ha, -1.0, Ha, nsums, x)
    e_u = u_a - uh_n
  else
    e_u = uh_n
  end

  cellfields = Vector{Pair}([
    "uh"=>uh_n,
    "e_u"=>e_u,
    "ph"=>ph_n,
    "jh"=>jh_n,
    "phi"=>φh_n,
    "div_jh"=>div_jh_n,
    "div_uh"=>div_uh_n,
  ])

  return cellfields, uh_0, kp
end

"""
  postprocess_3D(xh, model, Ω, b)

Post process operations and computations to be run after a 3D solution `xh` is
obtained, `Ω` is the `model`'s interior, and `b` the half-width in the direction
perpendicular to the external magnetic field.
"""
function postprocess_3D(xh, model, Ω, b)
  uh, ph, jh, φh = xh
  div_jh = ∇·jh
  div_uh = ∇·uh

  _isfluid = isfluid(b)
  dΩ = Measure(Ω, 6)
  uh_0 = sum(∫(uh*_isfluid)*dΩ)[3]/sum(∫(_isfluid)*dΩ)

  # Compute the dimensionless pressure drop gradient from the outlet value
  Grad_p = ∇·ph
  Γ = Boundary(model, tags="outlet")
  dΓ = Measure(Γ, 6)
  kp = sum(∫(-Grad_p*_isfluid)*dΓ)[3]/sum(∫(_isfluid)*dΓ)

  cellfields=[
    "uh"=>uh,
    "ph"=>ph,
    "jh"=>jh,
    "phi"=>φh,
    "div_uh"=>div_uh,
    "div_jh"=>div_jh,
    "grad_p"=>Grad_p,
  ]

  return cellfields, uh_0, kp
end
