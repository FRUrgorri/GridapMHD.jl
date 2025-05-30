"""
  channel_model(nc; b, L)
  
  )
  Function that returns an anonym function function that generates a Gridap model and the tags for BC. 
  Arguments are the geometrical characteristics and mesh characteristics of a rectangular cross-sectional channel.
#Arguments

-`nc: tuple of cells in each direction
-`b: channel aspect ratio
-`L: channel lenght
-`mesh_map = mesh map function

"""
function channel_model(nc::Tuple{Int64,Int64,Int64};
    b = 1,
    L = 2,
    mesh_map = nothing
    )
    
    domain = (-b, b, -1.0, 1.0, 0.0, L)
    
    function (parts,rank_partition)  
    model=CartesianDiscreteModel(parts, rank_partition, domain, nc; map=mesh_map)
  
    # Vertex tags: [1:8]
    # Edge tags: [9:20]
    # Surf tags: [21:26]
    labels = get_face_labeling(model)
    tags_inlet = append!(collect(1:4), [9, 10, 13, 14], [21])
    tags_outlet = append!(collect(5:8), [11, 12, 15, 16], [22])
    tags_insulated = append!(collect(1:20), [23, 24, 25, 26])
    add_tag_from_tags!(labels, "inlet", tags_inlet)
    add_tag_from_tags!(labels, "outlet", tags_outlet)
    add_tag_from_tags!(labels, "insulated", tags_insulated)
	
    #Neumann tags are the default, so there is no need to specified "outlet" as Neumann for example 
    Dirichlet_Utags=["inlet","insulated"]
    Dirichlet_Jtags=["inlet","outlet","insulated"]

    model, Dirichlet_Utags, Dirichlet_Jtags
    end
end

function channel_model(nc::Tuple{Int64,Int64};
    b = 1,
    mesh_map = nothing
    )
    
    domain = (-b, b, -1.0, 1.0, 0.0, 0.1)
    _nc = (nc...,3)

    function (parts,rank_partition)

    model = CartesianDiscreteModel(parts, rank_partition, domain, _nc; isperiodic=(false,false,true), map=mesh_map)
    
    # Vertex tags: [1:8]
    # Edge tags: [9:20]
    # Surf tags: [21:26]
    labels = get_face_labeling(model)
    tags_insulated = append!(collect(1:20), [23, 24, 25, 26])
    add_tag_from_tags!(labels, "insulated", tags_insulated)

    Dirichlet_Utags=["insulated"]
    Dirichlet_Jtags=["insulated"]

    model, Dirichlet_Utags, Dirichlet_Jtags
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
