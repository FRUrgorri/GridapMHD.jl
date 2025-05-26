function add_non_slip_at_solid_entity!(model,solid_entities,fluid_entity,name)
  D = num_cell_dims(model)
  labels = get_face_labeling(model)
  topo = get_grid_topology(model)
  cell_entity = get_cell_entity(labels)
  for d in 0:D-1  
    dface_entity = labels.d_to_dface_to_entity[d+1]
    dface_cells = get_faces(topo,d,D)
    cache = array_cache(dface_cells)
    for dface in 1:length(dface_cells)
      cells = getindex!(cache,dface_cells,dface)
      solid_found = false
      fluid_found = false
      for cell in cells
        solid_found = solid_found || cell_entity[cell] in solid_entities
        fluid_found = fluid_found || cell_entity[cell] == fluid_entity
      end
      if solid_found && fluid_found
        dface_entity[dface] = name
      end
    end
  end
end


"""
  stretchMHD(coord; domain, factor, dirs)

Particular case of a mesh stretching rule reproduced in [1] from the original source [2].

# Arguments
- `coord`: coordinate set describing the mesh to stretch.
- `domain`: domain over which the stretching is computed.
- `factor` : streching factor defined in [1].
- `dirs`: directions over which the stretching is computed.

[1]: S. Smolentsev et al., "Code development for analysis of MHD pressure drop reduction
in a liquid metal blanket using insulation technique based on a fully developed flow
model", Fusion Engineering and Design 73 (2005) 83-93.

[2]: G.O. Roberts, "Computational meshes for boundary layer problems", Proceedings of the
Second International Conference on Numerical Methods Fluid Dynamics, Lecture Notes on
Physics, vol. 8, Springer-Verlag, New York, 1971, pp. 171–177.
"""
function stretchMHD(
  coord;
  domain=(0.0,1.0,0.0,1.0,0.0,1.0),factor=(1.0,1.0,1.0),dirs=(1,2,3)
)
  ncoord = collect(coord.data)
  for (i,dir) in enumerate(dirs)
    ξ0 = domain[i*2-1]
    ξ1 = domain[i*2]
    l =  ξ1 - ξ0
    c = (factor[i] + 1)/(factor[i] - 1)

    if l > 0
      if ξ0 <= coord[dir] <= ξ1
        ξx = (coord[dir] - ξ0)/l                     # ξx from 0 to 1 uniformly distributed
        ξx_stretched = factor[i]*(c^ξx-1)/(1+c^ξx)    # ξx stretched from 0 to 1 towards 1
        ncoord[dir] =  ξx_stretched*l + ξ0            # coords stretched towards ξ1
      end
    else
      if ξ1 <= coord[dir] <= ξ0
        ξx = (coord[dir] - ξ0)/l                     # ξx from 0 to 1 uniformly distributed
        ξx_stretched = factor[i]*(c^ξx-1)/(1+c^ξx)    # ξx stretched from 0 to 1 towards 1
        ncoord[dir] =  ξx_stretched*l + ξ0            # coords stretched towards ξ1
      end
    end
  end
  return VectorValue(ncoord)
end

"""
  ChangeDensity(coord;domain,subDomain, nodesTo, nodesSub, dirs)
  
  Function that redistribute the total number of cells into two different regions
  
# Arguments
- `coord` :  coordinate set describing the mesh. 
- `domain` : tuple with the edges of the total geometrical domain along `dir` direction
- `subdomain` : tuple with the edges of the geometrical subdomain inside `domain` where the cell density will be changed
- `cellsTot` : tuple with total number of cells in the domain alog `dir` 
- `cellsSub` : tuple with the number of cells in the subdomain along `dir`. It has to be lower than `cellTot`
- `dirs`: tuple with directions over which the density change is computed.
 
  
"""
function ChangeDensity(coord;domain=(0.0,1.0,0.0,1.0,0.0,1.0),subDomain=(0.0,1.0,0.0,1.0,0.0,1.0),
			     cellsTot=(1.0,1.0,1.0), cellsSub=(1.0,1.0,1.0), dirs=(1,2,3))
  ncoord = collect(coord.data)
  for (i,dir) in enumerate(dirs)
    ξ0 = domain[i*2-1]
    ξ1 = domain[i*2]
    Ltot =  ξ1 - ξ0
    Lsub = subDomain[i*2] - subDomain[i*2-1]
    @assert Ltot > Lsub
    
    alpha = (Lsub/Ltot)*(cellsTot[i]/cellsSub[i])
    betta = ((Ltot-Lsub)/Ltot)*(cellsTot[i]/(cellsTot[i]-cellsSub[i]))

    if Ltot != Lsub
      if ξ0 <= coord[dir] <= ξ1
        if coord[dir] <= (Lsub/alpha + ξ0)
          ncoord[dir] = alpha*coord[dir] + ξ0*(1-alpha)
        else
          ncoord[dir] = betta*coord[dir] + ξ0*(1-betta) + Lsub*(1-betta/alpha)
        end
      end
    end
  end
  return VectorValue(ncoord)
end
