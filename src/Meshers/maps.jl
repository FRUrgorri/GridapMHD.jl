"""
    map_Roberts(b,Ha)
    
Function that returns a function for the cross sectional map using Roberts formula.
See strechMHD function for detailed formula

# Arguments
- `b` :  Channel aspect ratio
- `Ha: Streching factor along the direction perpendicular to B
    
"""
function map_Roberts(b,Ha)
     
     stretch_Ha = sqrt(Ha/(Ha-1))
     stretch_side = sqrt(sqrt(Ha)/(sqrt(Ha)-1))
     
     function (coord)
       ncoord = stretchMHD(coord,domain=(0,-b,0,-1.0),factor=(stretch_side,stretch_Ha),dirs=(1,2))
       ncoord = stretchMHD(ncoord,domain=(0,b,0,1.0),factor=(stretch_side,stretch_Ha),dirs=(1,2))
       ncoord
     end
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
