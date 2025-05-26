
function hunt_stretch_map(
  L::Real,Ha::Real,kmap_x::Number,kmap_y::Number,BL_adapted::Bool
)
  stretch_Ha = sqrt(Ha/(Ha-1))
  stretch_side = sqrt(sqrt(Ha)/(sqrt(Ha)-1))
  function map1(x)
    y = stretchMHD(x,domain=(0,-L,0,-L),factor=(stretch_side,stretch_Ha),dirs=(1,2))
    z = stretchMHD(y,domain=(0,L,0,L),factor=(stretch_side,stretch_Ha),dirs=(1,2))
    return z  
  end
  function map2(x)
    layer(x,a) = sign(x)*abs(L*x)^(1/a)
    return VectorValue(layer(x[1],kmap_x),layer(x[2],kmap_y),x[3])
  end
  coord_map = BL_adapted ? map1 : map2
  return coord_map
end

function hunt_add_tags!(model::GridapDistributed.DistributedDiscreteModel,L::Real,tw::Real)
  map(local_views(model)) do model
    hunt_add_tags!(model,L,tw)
  end
end

function hunt_add_tags!(model,L::Real,tw::Real)
  labels = get_face_labeling(model)
  if tw > 0.0 ## add solid tags
    # When model is part of a distributed model we are using
    # that the maximum entity (9) is the same in all parts, 
    # which is true for a GenericDistributedDiscreteModel
    # A reduction would be needed in general
    # cell_entity = labels.d_to_dface_to_entity[end]
    cell_entity = get_cell_entity(labels)
    solid_1 = maximum(cell_entity) + 1
    solid_2 = solid_1 + 1
    fluid = solid_2 + 1
    noslip = fluid + 1
    function set_entities(xs)
      tol = 1.0e-9
      if all(x->(x[1]>L-tol)||x[1]<-L+tol,xs)
        solid_1
      elseif all(x->(x[2]>L-tol)||x[2]<-L+tol,xs)
        solid_2
      else
        fluid
      end
    end
    grid = get_grid(model)
    cell_coords = get_cell_coordinates(grid)
    copyto!(cell_entity,map(set_entities,cell_coords))
    add_tag!(labels,"solid_1",[solid_1])
    add_tag!(labels,"solid_2",[solid_2])
    add_tag!(labels,"solid",[solid_1,solid_2])
    add_tag!(labels,"fluid",[fluid])
    tags_j = vcat(collect(1:(8+12)),collect((1:4).+(8+12+2)))
    add_tag_from_tags!(labels,"insulating",tags_j)
    add_non_slip_at_solid_entity!(model,[solid_1,solid_2],fluid,noslip)
    add_tag!(labels,"noslip",[noslip])
  else    
    tags_u = append!(collect(1:20),[23,24,25,26])
    tags_j = append!(collect(1:20),[25,26])
    add_tag_from_tags!(labels,"noslip",tags_u)
    add_tag_from_tags!(labels,"insulating",tags_j)
  end
end

"""
    hunt_generate_base_mesh(ranks,nc::Tuple,np::Tuple,L::Real,Ha::Real,kmap_x::Number,kmap_y::Number,BL_adapted::Bool)

  Generate a mesh for the Hunt problem, distributed amongst the communicator `ranks`.
"""
function hunt_generate_base_mesh(
  ranks,np::Tuple,
  nc::Tuple,L::Real,tw::Real,Ha::Real,kmap_x::Number,kmap_y::Number,BL_adapted::Bool
)
  Lt = L+tw
  coord_map = hunt_stretch_map(Lt,Ha,kmap_x,kmap_y,BL_adapted)
  _nc = (nc[1],nc[2],3)
  _np = (np[1],np[2],1)
  domain = (-1.0,1.0,-1.0,1.0,0.0,0.1)
  model = CartesianDiscreteModel(ranks,_np,domain,_nc;isperiodic=(false,false,true),map=coord_map)
  hunt_add_tags!(model,L,tw)
  return model
end

"""
    hunt_generate_base_mesh(nc::Tuple,np::Tuple,L::Real,Ha::Real,kmap_x::Number,kmap_y::Number,BL_adapted::Bool)

  Generate a serial mesh for the Hunt problem.
"""
function hunt_generate_base_mesh(
  nc::Tuple,L::Real,tw::Real,Ha::Real,kmap_x::Number,kmap_y::Number,BL_adapted::Bool
)
  Lt = L+tw
  coord_map = hunt_stretch_map(Lt,Ha,kmap_x,kmap_y,BL_adapted)
  _nc = (nc[1],nc[2],3)
  domain = (-1.0,1.0,-1.0,1.0,0.0,0.1)
  model = CartesianDiscreteModel(domain,_nc;isperiodic=(false,false,true),map=coord_map)
  hunt_add_tags!(model,L,tw)
  return model
end
