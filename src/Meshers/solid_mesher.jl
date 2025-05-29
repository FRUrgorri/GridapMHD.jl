"""
  channel_model(;
  
  )
  Function that returns an anonym function function that generates a Gridap model. Arguments are the geometrical 
  characteristics and mesh characteristics of a rectangular cross-sectional channel.
#Arguments

-`Ha` : Hartmann number
-`b`: channel aspect ratio
-`L`: channel lenght
-`nc`: tuple of cells in each direction

"""

function channel_model(nc::Tuple{Int64,Int64,Int64};
    Ha = 10,
    b = 1,
    L = 2 
    )
    
    domain = (-b, b, -1.0, 1.0, 0.0, L)
    
    stretch_Ha = sqrt(Ha/(Ha-1))
    strech_side = sqrt(sqrt(Ha)/(sqrt(Ha)-1))
    mesh_map(coord) = map_cross_section(coord,b, stretch_Ha,strech_side)
    
    function (parts,rank_partition)  
    model=CartesianDiscreteModel(parts, rank_partition, domain, nc; map=mesh_map)
  # Vertex tags: [1:8]
  # Edge tags: [9:20]
  # Surf tags: [21:26]
    labels = get_face_labeling(model)
    tags_inlet = append!(collect(1:4), [9, 10, 13, 14], [21])
    tags_outlet = append!(collect(5:8), [11, 12, 15, 16], [22])
    tags_insulated = append!(collect(1:20), [23, 24, 25, 26])
    tags_noslip=tags_insulated
    add_tag_from_tags!(labels, "inlet", tags_inlet)
    add_tag_from_tags!(labels, "outlet", tags_outlet)
    add_tag_from_tags!(labels, "insulated", tags_insulated)
    add_tag_from_tags!(labels, "noslip", tags_noslip)
    
    model
    end
end

function channel_model(nc::Tuple{Int64,Int64};
    Ha = 10,
    b = 1
    )
    
    domain = (-b, b, -1.0, 1.0, 0.0, 0.1)
    
    stretch_Ha = sqrt(Ha/(Ha-1))
    strech_side = sqrt(sqrt(Ha)/(sqrt(Ha)-1))
    mesh_map(coord) = map_cross_section(coord,b, stretch_Ha,strech_side)
    
    function (parts,rank_partition) 
    model = CartesianDiscreteModel(parts, rank_partition, domain, nc; isperiodic=(false,false,true), map=mesh_map)
    
    labels = get_face_labeling(model)
    tags_inlet = 0
    tags_outlet = 0
    tags_insulated = append!(collect(1:20), [23, 24, 25, 26])
    tags_noslip=tags_insulated
    add_tag_from_tags!(labels, "inlet", tags_inlet)
    add_tag_from_tags!(labels, "outlet", tags_outlet)
    add_tag_from_tags!(labels, "insulated", tags_insulated)
    add_tag_from_tags!(labels, "noslip", tags_noslip)
   
    model
    end
end


function channel_model(nl::Tuple{Int64,Int64},ns::Tuple{Int64,Int64},nz::Int64;
    Ha = 10,
    b = 1,
    L = 2,
    tw_Ha = 0,
    tw_s = 0,
    cw_Ha = 0.0,
    cw_s = 0.0, 
    fluid_stretching = :Roberts,
    fluid_stretch_params = (0.5, 1.0)
)
"""
   The total number of mesh elements is the liquid region elements plus
   ns elements in the direction of the solid domain.
   2*ns accounts for ns solid cells on each side of the liquid for each
   direction
   nL nodes in the flow direction
""" 
  nc = ((nl .+ (2 .* ns))...,nz)
  
  
   # Domain definitions
  domain_liq = (-b, b, -1.0, 1.0, 0.0, L)
  domain = domain_liq .+ (-tw_s, tw_s, -tw_Ha, tw_Ha, 0.0, 0.0)
  
   #Computation of the mesh map 
  mesh_map = solid_mesh_map(
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

   (parts,rank_partition) -> CartesianDiscreteModel(parts, rank_partition, domain, nc; map=mesh_map)
end

function channel_model(nl::Tuple{Int64,Int64},ns::Tuple{Int64,Int64};
    Ha = 10,
    b = 1,
    tw_Ha = 0,
    tw_s = 0,
    cw_Ha = 0.0,
    cw_s = 0.0, 
    fluid_stretching = :Roberts,
    fluid_stretch_params = (0.5, 1.0)
)
"""
   The total number of mesh elements is the liquid region elements plus
   ns elements in the direction of the solid domain.
   2*ns accounts for ns solid cells on each side of the liquid for each
   direction
   nL nodes in the flow direction
""" 
  nc = ((nl .+ (2 .* ns))...,3)
  
  
   # Domain definitions
  domain_liq = (-b, b, -1.0, 1.0, 0.0, 0.1)
  domain = domain_liq .+ (-tw_s, tw_s, -tw_Ha, tw_Ha, 0.0, 0.0)
  
   #Computation of the mesh map 
  mesh_map = solid_mesh_map(
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

   (parts,rank_partition) -> CartesianDiscreteModel(parts, rank_partition, domain, nc; map=mesh_map, isperiodic=(false,false,true))
end

"""
    map_cross_section(coord,b,strech_side,strech_Ha)
    
Function that applies a cross sectional map using Roberts formula.
See strechMHD function

# Arguments
- `coord`: Original coordinates (uniformly distributed)
- `b` :  Channel aspect ratio
- `strech_Ha: Streching factor along the direction parallel to B
- `strech_side: Streching factor along the direction perpendicular to B
    
"""
function map_cross_section(coord,b,strech_Ha,strech_side)
     ncoord = stretchMHD(coord,domain=(0,-b,0,-1.0),factor=(strech_side,strech_Ha),dirs=(1,2))
     ncoord = stretchMHD(ncoord,domain=(0,b,0,1.0),factor=(strech_side,strech_Ha),dirs=(1,2))
     ncoord  
end

"""
  solid_mesh_map(
    Ha, b, tw_Ha, tw_s, nc, nl, ns, domain, fluid_stretching, fluid_stretch_params
  )

Function that returns a map function to pass to `CartesianDiscreteModel` defining
the mesh stretching to accomodate more nodes towards the boundary layers, and to
define a uniform mesh on the solid region.  The returned function depends only of
one argument (the coordinates) as expected by `CartesianDiscreteModel`.

# Arguments
- `b`: half-width in the direction perpendicular to the external magnetic field.
- `tw_Ha`: width of the solid wall in the external magnetic field direction.
- `tw_s`: width of the solid wall normal to the external magnetic field.
- `nc`: array containing the total number of cells.
- `nl`: array containing the number of cells in the fluid region.
- `ns`: array containing the number of cells in the solid region.
- `domain`: array describing the computational domain: (x0, xf, y0, yf, z0, zf).
- `fluid_stretching`: stretching rule.
- `fluid_stretch_params`: stretching parameters (rule dependent).
"""
function solid_mesh_map(
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
  function _mesh_map(coord)
    if (tw_s > 0.0) || (tw_Ha > 0.0)
      coord = solidMap(coord, tw_Ha, tw_s, nc, ns, nl, domain)
    end

    if fluid_stretching == :uniform
    elseif fluid_stretching == :Roberts
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
        coord = stretchMHD(coord, domain=(0, -1.0), factor=(stretch_Ha,), dirs=(2,))
        coord = stretchMHD(coord, domain=(0, 1.0), factor=(stretch_Ha,), dirs=(2,))
      end
    elseif fluid_stretching == :hyperbolic
      coord = stretch_map(coord, x_hyp, b, 1)
      coord = stretchMHD(coord, domain=(0, -1.0), factor=(stretch_Ha,), dirs=(2,))
      coord = stretchMHD(coord, domain=(0, 1.0), factor=(stretch_Ha,), dirs=(2,))
    end

    return coord
  end

  @assert fluid_stretching ∈ (:uniform, :Roberts, :hyperbolic)
  stretch_Ha = sqrt(Ha/(Ha-1))
  if fluid_stretching == :Roberts
    @assert length(fluid_stretch_params) == 2
    γ, δ = fluid_stretch_params
    stretch_γ = 1/sqrt(1 - δ/Ha^γ)
  elseif fluid_stretching == :hyperbolic
    @assert length(fluid_stretch_params) == 2
    l_BL, n_δ = fluid_stretch_params
    δ = 1/sqrt(Ha)
    x_hyp = map_hyp(b, l_BL, δ, nl[1]; n_δ=n_δ)
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
    if nx < ns[1]stretch_Ha
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

"""
  map_hyp(xf, l_BL, δ, nl; n_δ=2.0)

Hyperbolic element distribution aimed for the liquid region.

The aim is to distribute the mesh elements so that their separation follows a tanh rule
with a minimum near the fluid-solid boundary (given by `xf`).  `l_BL` determines the initial
element separation; `δ` determines where the inflection point lies; `nl` gives the number of
elements; and the keyword argument `n_δ` can modify the slope of the separation distribution.
"""
function map_hyp(xf::Real, l_BL::Real, δ::Real, nl::Real; n_δ=2.0)
  function _l(x)
    cell_length = (l_core - l_BL)/2*(1 - tanh((x - xf + n_δ*δ)/δ)) + l_BL

    return cell_length
  end

  n = ceil(Int, nl/2)
  l_core = 2*(xf - n*l_BL)/n + l_BL  # Initial guess

  xnew = Vector{Float64}(undef, n)
  xnew[end] = xf
  for i in 1:(n - 1)
    xnew[end - i] = xnew[end - i + 1] - _l(xnew[end - i + 1])
  end

  l_core_final = (xnew[2] - xnew[1])/(xf - xnew[1])  # Final guess
  if iseven(nl)
    # Renormalize so that xnew[1] = l_core
    Δx₀ = l_core_final
  else
    # Renormalize so that xnew[1] = l_core/2
    Δx₀ = l_core_final/2
  end
  xnew = ((xnew .- xnew[1])./(xf - xnew[1]) .+ Δx₀)./(xf + Δx₀)

  return xnew
end

"""
  stretch_map(coord, xnew, xf, dir)

Stretch the mesh symmetrically following the distribution given by `xnew` in the region
`(-xf, xf)` along the `dir` direction.  `xnew` is assumed to give the `(0, xf)`
distribution.
"""
function stretch_map(coord, xnew, xf::Real, dir::Integer)
  n = length(xnew)
  ncoord = collect(coord.data)
  if 0.0 < coord[dir] <= xf
    ncoord[dir] = xnew[round(Int, coord[dir]*n/xf)]
  elseif -xf <= coord[dir] < 0.0
    ncoord[dir] = -xnew[round(Int, abs(coord[dir])*n/xf)]
  end

  return VectorValue(ncoord)
end

"""
  solid_add_tags!(model, b::Real, tw_Ha::Real, tw_s::Real)
  
Assign tags to entities for every possible combination of tw_Ha and tw_s.
In the process, it identifies the mesh elements corresponding to the solid
region.
""" 
function solid_add_tags!(
  model::GridapDistributed.DistributedDiscreteModel,
  b::Real,
  tw_Ha::Real,
  tw_s::Real,
)   
  map(local_views(model)) do model
    solid_add_tags!(model, b, tw_Ha, tw_s)
  end
    
  return nothing
end 
    
function solid_add_tags!(model, b::Real, tw_Ha::Real, tw_s::Real)
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
    # The actual fluid-solid boundary is tagged w/ solid_wall_BC()
    add_tag_from_tags!(labels, "fluid-solid-boundary", Vector{Int}())
  end

  return nothing
end

"""
  solid_wall_BC(cw_Ha::Real, cw_s::Real, tw_Ha::Real, tw_s::Real)

Selects the insulating BC or thin wall BC for both Ha and Side walls according
to cw and tw.
    
Also selects the tags for the no-slip velocity BC which do not lie in a
fluid-solid boundary, i.e., when tw=0.0 in that side.
""" 
function solid_wall_BC(
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
