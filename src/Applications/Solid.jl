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
- `B_var = :uniform`: external magnetic field form.
- `B_coef = nothing`: coefficients describing the external magnetic field function.
- `dir_B = (0.0,1.0,0.0)`: external magnetic field direction vector.
- `b = 1.0`: half-width in the direction perpendicular to the external magnetic field.
- `L = nothing`: length in the axial direction.
- `tw_Ha = 0.0`: width of the solid wall in the external magnetic field direction.
- `tw_s = 0.0`: width of the solid wall normal to the external magnetic field.
- `cw_Ha = 0.0`: wall parameter in the external magnetic field direction.
- `cw_s = 0.0`: wall parameter normal to the external magnetic field.
- `inlet = nothing`: velocity inlet boundary condition function.
- `vtk = true`: toggle to save the final results in vtk format.
- `solve = true`: toggle to run the solver.
- `solver = :julia`: solver to be used and additional solver parameters.
- `verbose = true`: print time statistics.
- `mesh2vtk = false`: save the generated model in vtk format.
- `stretch_γ = 0.5`: stretching factor for the Side boundary layer.
- `τ_Ha = 100.0`: penalty term for the thin wall boundary condition in the Ha boundary.
- `τ_s = 100.0`: penalty term for the thin wall boundary condition in the Side boundary.
- `res_assemble = false`: toggle to time the computation of the residual independently.
- `jac_assemble = false`: toggle to time the computation of the jacobian independently.
- `nsums = 10`: number of terms used for the analytical solution computation.

# Fully developed approximation
For the fully developed approximation, the following arguments need be set:
- `nc`, and `np` must be 2-dimensional arrays.
- `L`, `inlet`, and `N` must be set to `nothing`.
- `Re` must be set to `1.0`.

# 3D simulation
If a full 3D simulation is to be run, the following arguments need be set:
- `nc`, and `np` must be 3-dimensional arrays.
- `L`, and `N` must be numbers.
- `inlet` must be set to one of the available boundary condition keys (see #Inlet).
- `Re` must be set to `nothing`.

# Inlet
Available keys for the inlet velocity boundary condition:
- `:uniform`: uniform value in the cross section.
- `:parabolic`: parabolic profile of a fully developed flow.

# External magnetic field
Available forms for the external magnetic field (`B_var`):
- `:uniform`:
  uniform value in the whole computational domain.
  `B_coef` is thus ignored.
- `:polynomial`:
  external magnetic field magnitude varies along the axial direction following a
  polynomic function.
  `B_coef` determines the coefficients of said polynomial in increasing order.
"""
function Solid(;
  backend = nothing,
  np = nothing,
  title = "Solid",
  nruns = 1,
  path = ".",
  kwargs...)

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
  B_var = :uniform,
  B_coef = nothing,
  dir_B = (0.0,1.0,0.0),
  b = 1.0,
  L = nothing,
  tw_Ha = 0.0,
  tw_s = 0.0,
  cw_Ha = 0.0,
  cw_s = 0.0,
  inlet = nothing,
  vtk = true,
  solve = true,
  solver = :julia,
  verbose = true,
  mesh2vtk = false,
  stretch_γ = 0.5,
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

  dirB = (1/norm(VectorValue(dir_B)))*VectorValue(dir_B)

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
    isa(inlet, Nothing) && (Re == 1.0) && isa(N, Nothing)
  )
  Full3D = (
    (length(nc) == 3) && (length(rank_partition) == 3) && isa(L, Number) &&
    !isa(inlet, Nothing) && isa(Re, Nothing) && isa(N, Number)
  )
  if FD
    _nc = (nc[1],nc[2],3)
    _rank_partition = (rank_partition[1], rank_partition[2], 1)
    L = 0.1
    N = Ha^2/Re
    f = VectorValue(0.0, 0.0, 1.0)
    periodic = (false, false, true)
    Bfield = dirB
  elseif Full3D
    _nc = nc
    _rank_partition = rank_partition
    Re = Ha^2/N
    f = VectorValue(0.0, 0.0, 0.0)
    periodic = (false, false, false)
    if ns[3] > 0
      error("No solid elements allowed at inlet/outlet regions.")
    end
    if inlet == :parabolic
      u_inlet((x,y,z)) = VectorValue(0, 0, (9/(4*b^3))*(x^2 - b^2)*(y^2 - 1))
    elseif inlet == :uniform
      u_inlet = VectorValue(0.0, 0.0, 1.0)
    end
    if B_var == :uniform
      Bfield = dirB
    elseif B_var == :polynomial
      # B_coef assumed to be normalized w.r.t. given Ha
      Bfield = x -> dirB .* sum(B_coef .* [x[3]^(i-1) for i in 1:length(B_coef)])
    elseif B_var == :tanh
      # B_coef[1] roughly determines the length over which B > 0
      Bfield = x -> dirB*(0.5*(1 - tanh(abs(2*x[3]/B_coef[1] - 1)/0.1 - 5)))
    else
      error("Unrecognized magnetic field input.")
    end
  else
    error(
      "Input args are not compatible with either a 3D computation or a \
      fully developed approximation."
    )
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
  _mesh_map = mesh_map(Ha, b, tw_Ha, tw_s, nc, nl, ns, domain, stretch_γ)
  model = CartesianDiscreteModel(
    parts, _rank_partition, domain, _nc;
    isperiodic=periodic, map=_mesh_map
  )
  solidFD_add_tags!(model, b, tw_Ha, tw_s)
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
  insulated_tags, thinWall_params, noslip_extra_tags = wall_BC(model, cw_Ha, cw_s, tw_Ha, tw_s, τ_Ha, τ_s)
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
  θⱼ = acosd(dirB·VectorValue(0.0, 1.0, 0.0))

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

  if vtk
    if (tw_Ha > 0.0) && (tw_s > 0.0)
      push!(cellfields, "σ"=>σ_Ω)
    end
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
  info[:inlet] = inlet
  info[:B_var] = B_var
  info[:B_coef] = B_coef
  info[:theta_y] = θⱼ

  return info, t
end


# Conductivity to CellField
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

# Tag related functions
"""
  solidFD_add_tags!(model, b::Real, tw_Ha::Real, tw_s::Real)

Assign tags to entities for every possible combination of tw_Ha and tw_s.
In the process, it identifies the mesh elements corresponding to the solid
region.
"""
function solidFD_add_tags!(
  model::GridapDistributed.DistributedDiscreteModel,
  b::Real,
  tw_Ha::Real,
  tw_s::Real,
)
  map(local_views(model)) do model
    solidFD_add_tags!(model, b, tw_Ha, tw_s)
  end

  return nothing
end

function solidFD_add_tags!(model, b::Real, tw_Ha::Real, tw_s::Real)
  labels = get_face_labeling(model)
  tags_inlet = append!(collect(1:20),[21])
  tags_outlet = append!(collect(1:20),[22])
  # These are the external walls, i.e., not necesarily the fluid-solid boundary
  tags_j_Ha = append!(collect(1:20), [23,24])
  tags_j_side = append!(collect(1:20), [25,26])
  add_tag_from_tags!(labels, "inlet", tags_inlet)
  add_tag_from_tags!(labels, "outlet", tags_outlet)
  add_tag_from_tags!(labels, "Ha_ext_walls", tags_j_Ha)
  add_tag_from_tags!(labels, "side_ext_walls", tags_j_side)

  if (tw_Ha > 0.0) || (tw_s > 0.0)
    cell_entity = get_cell_entity(labels)
    # Label numbering
    if (tw_Ha > 0.0) && (tw_s == 0.0)
      solid_Ha = maximum(cell_entity) + 1
      solid_s = nothing
      fluid = solid_Ha + 1
    elseif (tw_s > 0.0) && (tw_Ha == 0.0)
      solid_Ha = nothing
      solid_s = maximum(cell_entity) + 1
      fluid = solid_s + 1
    else
      solid_Ha = maximum(cell_entity) + 1
      solid_s = solid_Ha + 1
      fluid = solid_s + 1
    end
    noslip = fluid + 1

    function set_entities(xs)
      tol = 1.0e-9
      if all(x->(x[1]>b-tol)||x[1]<-b+tol,xs)
        solid_s
      elseif all(x->(x[2]>b-tol)||x[2]<-b+tol,xs)
        solid_Ha
      else
        fluid
      end
    end

    grid = get_grid(model)
    cell_coords = get_cell_coordinates(grid)
    copyto!(cell_entity, map(set_entities, cell_coords))
    solid_entities = Vector{Int}()
    if !isnothing(solid_Ha)
      add_tag!(labels, "solid_Ha", [solid_Ha])
      push!(solid_entities, solid_Ha)
    end
    if !isnothing(solid_s)
      add_tag!(labels, "solid_s", [solid_s])
      push!(solid_entities, solid_s)
    end
    add_tag!(labels, "solid", solid_entities)
    add_tag!(labels, "fluid", [fluid])
    Meshers.add_non_slip_at_solid_entity!(model, solid_entities, fluid, noslip)
    add_tag!(labels, "fluid-solid-boundary", [noslip])
  else
    # If both walls are missing, "fluid-solid-boundary" tag is created empty
    # The actual fluid-solid boundary is tagged w/ wall_BC()
    add_tag_from_tags!(labels, "fluid-solid-boundary", Vector{Int}())
  end

  return nothing
end

"""
  wall_BC(cw_Ha::Real, cw_s::Real, tw_Ha::Real, tw_s::Real)

Selects the insulating BC or thin wall BC for both Ha and Side walls according
to cw and tw.

Also selects the tags for the no-slip velocity BC which do not lie in a
fluid-solid boundary, i.e., when tw=0.0 in that side.
"""
function wall_BC(
  model::GridapDistributed.DistributedDiscreteModel,
  cw_Ha::Real,
  cw_s::Real,
  tw_Ha::Real,
  tw_s::Real,
  τ_Ha::Real,
  τ_s::Real
)
  function _wall_BC!(cw::Real, tw::Real, τ::Real, tag::String)
    if (cw > 0.0) && (tw == 0.0)
      push!(
        thinWall_options, Dict(
          :cw => cw,
          :τ => τ,
          :domain =>Boundary(model, tags=tag)
        )
      )
    else
      push!(insulated_tags, tag)
    end

    if tw == 0.0
      # add_non_slip_at_solid_entity! does not detect this tag as a fluid-solid
      # boundary, so it must be manually added to the no-slip BC tag list
      push!(noslip_extra_tags, tag)
    end

    return nothing
  end

  insulated_tags = Vector{String}()
  thinWall_options = Vector{Dict{Symbol,Any}}()
  noslip_extra_tags = Vector{String}()

  _wall_BC!(cw_Ha, tw_Ha, τ_Ha, "Ha_ext_walls")
  _wall_BC!(cw_s, tw_s, τ_s, "side_ext_walls")

  return insulated_tags, thinWall_options, noslip_extra_tags
end

# Mesher maps and other helper funcs
"""
  mesh_map(coord)

Map function to pass to CartesianDiscreteModel.

Combines the solid mesh map and the liquid stretchMHD map as required depending
on the input tw_s, tw_Ha values and the stretch_γ exponent.

stretch_γ determines the stretching in the Side boundary layer.
"""
function mesh_map(Ha, b, tw_Ha, tw_s, nc, nl, ns, domain, γ)
  function _mesh_map(coord)
    stretch_Ha = sqrt(Ha/(Ha-1))
    stretch_γ = sqrt(Ha^γ/(Ha^γ-1))

    if (tw_s > 0.0) || (tw_Ha > 0.0)
      coord = solidMap(coord, tw_Ha, tw_s, nc, ns, nl, domain)
    end

    if γ > 0.0
      coord = stretchMHD(
        coord,
        domain=(0, -b, 0, -1.0),
        factor=(stretch_γ, stretch_Ha),
        dirs=(1, 2),
      )
      coord = stretchMHD(
        coord,
        domain=(0, b, 0, 1.0),
        factor=(stretch_γ, stretch_Ha),
        dirs=(1, 2),
      )
    else
      coord = stretchMHD(
        coord,
        domain=(0, -1.0),
        factor=(stretch_Ha,),
        dirs=(2,),
      )
      coord = stretchMHD(
        coord,
        domain=(0, 1.0),
        factor=(stretch_Ha),
        dirs=(2,),
      )
    end

    return coord
  end

  return _mesh_map
end

"""
  solidMap(coord, tw_Ha, tw_s, nc, ns, nl, domain)

Map to space out the mesh nodes evenly in the solid region according to the
specified number of solid cells, ns, and the solid width, tw.  It also requires
the total number of cells, nc, the number of cells in the liquid region, nl, and
the domain.

tw_s and tw_Ha are the side wall and Hartmann wall respective thickness, i.e.,
they are the solid width in the X and Y directions, respectively.

nc, ns, and nl are assumed to be arrays corresponding to the X and Y directions
respectevely.

domain is an array specifying the domain limits:
  domain = (x0, xf, y0, yf,...)
"""
function solidMap(coord, tw_Ha, tw_s, nc, ns, nl, domain)
  ncoord = collect(coord.data)
  x0 = domain[1]
  xf = domain[2]
  y0 = domain[3]
  yf = domain[4]
  dx = (xf - x0)/nc[1]
  dxl = (xf - x0 - 2*tw_s)/nl[1]
  dxs = tw_s/ns[1]
  dy = (yf - y0)/nc[2]
  dyl = (yf - y0 - 2*tw_Ha)/nl[2]
  dys = tw_Ha/ns[2]

  if tw_s > 0.0
    nx = abs(ncoord[1] - x0)/dx
    if nx < ns[1]
      ncoord[1] = x0 + nx*dxs
    elseif ns[1] <= nx <= (nl[1] + ns[1])
      ncoord[1] = x0 + tw_s + (nx - ns[1])*dxl
    elseif nx > (nl[1] + ns[1])
      ncoord[1] = x0 + tw_s + nl[1]*dxl + (nx - nl[1] - ns[1])*dxs
    end
  end
  if tw_Ha > 0.0
    ny = abs(ncoord[2] - y0)/dy
    if ny < ns[2]
      ncoord[2] = y0 + ny*dys
    elseif ns[2] <= ny <= (nl[2] + ns[2])
      ncoord[2] = y0 + tw_Ha + (ny - ns[2])*dyl
    elseif ny > (nl[2] + ns[2])
      ncoord[2] = y0 + tw_Ha + nl[2]*dyl + (ny - nl[2] - ns[2])*dys
    end
  end
  return VectorValue(ncoord)
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

"""
  isfluid(X)

Returns 1.0 if coordinates given by the three-component vector X fall within the
fluid region or 0.0 otherwise.

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
obtained.  `Ω` is the `model`'s interior.
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
  kp_a = nothing
  dev_kp = nothing

  cellfields=[
    "uh"=>uh,
    "ph"=>ph,
    "jh"=>jh,
    "phi"=>φh,
    "div_uh"=>div_uh,
    "div_jh"=>div_jh,
    "grad_p"=>Grad_p,
  ]

  return cellfields, uh_0, kp, kp_a, dev_kp
end

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
