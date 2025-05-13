##Functions for mesh manipulation

function ChangeDensity(coord;domain=(0.0,1.0,0.0,1.0,0.0,1.0),subDomain=(0.0,1.0,0.0,1.0,0.0,1.0),
			     nodesTot=(1.0,1.0,1.0), nodesSub=(1.0,1.0,1.0), dirs=(1,2,3))
  ncoord = collect(coord.data)
  for (i,dir) in enumerate(dirs)
    ξ0 = domain[i*2-1]
    ξ1 = domain[i*2]
    Ltot =  ξ1 - ξ0
    Lsub = subDomain[i*2] - subDomain[i*2-1]

    alpha = (Lsub/Ltot)*(nodesTot[i]/nodesSub[i])
    betta = ((Ltot-Lsub)/Ltot)*(nodesTot[i]/(nodesTot[i]-nodesSub[i]))

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

##Functions for geometry generation

function cuboid(;dx=1,dy=1,dz=1,x0=Point(0,0,0),name="cuboid",faces=["face$i" for i in 1:6])
  e1 = VectorValue(1,0,0)
  e2 = VectorValue(0,1,0)
  e3 = VectorValue(0,0,1)

  plane1 = plane(x0=x0-0.5*dz*e3,v=-e3,name=faces[1])
  plane2 = plane(x0=x0+0.5*dz*e3,v=+e3,name=faces[2])
  plane3 = plane(x0=x0-0.5*dy*e2,v=-e2,name=faces[3])
  plane4 = plane(x0=x0+0.5*dy*e2,v=+e2,name=faces[4])
  plane5 = plane(x0=x0-0.5*dx*e1,v=-e1,name=faces[5])
  plane6 = plane(x0=x0+0.5*dx*e1,v=+e1,name=faces[6])

  geo12 = intersect(plane1,plane2)
  geo34 = intersect(plane3,plane4)
  geo56 = intersect(plane5,plane6)

  return intersect(intersect(geo12,geo34),geo56,name=name)
end

# Function that given a filter set a name to those elements that satisfy the
# conditions of the filter
function add_entity!(model,in,name)
  labels = get_face_labeling(model)
  node_coordinates = get_node_coordinates(model)
  entity = num_entities(labels) + 1
  for d in 0:num_dims(model)-1
    facets = get_face_nodes(model,d)
    for (i,facet) in enumerate(facets)
      coord = sum(node_coordinates[facet])/length(facet)
      if in(coord)
        labels.d_to_dface_to_entity[d+1][i] = entity
      end
    end
  end
  add_tag!(labels,name,[entity])
end

# Analytical formulas for pressure drop gradients

function kp_shercliff_cartesian(b,Ha)
  kp = 1/(Ha*(1-0.852*Ha^(-0.5)/b-1/Ha))
kp
end

function kp_shercliff_cylinder(Ha)
  kp = (3/8)*pi/(Ha-(3/2)*pi)
kp
end

function kp_hunt(b,Ha)
  kp = 1/(Ha*(1-0.956*Ha^(-0.5)/b-1/Ha))
kp
end

function kp_tillac(b,Ha,cw_s,cw_Ha)
  k_s = (1/(3*b))*(Ha^(0.5)/(1+cw_s*Ha^(0.5)))
  k_Ha = (1+cw_Ha)/(1/Ha + cw_Ha)
  kp = 1/(k_s+k_Ha)
kp
end

function kp_glukhih(Ha,cw)
  kp = (3/8)*pi*(1+0.833*cw*Ha-0.019*(cw*Ha)^2)/Ha
kp
end

kp_Miyazaki_circular(cw) = cw/(cw + 1)

kp_Miyazaki_rectangular(cw, a, b) = cw/(1 + a/(3*b) + cw)

kp_Miyazaki_rectangular(cw, b) = kp_Miyazaki_rectangular(cw, 1.0, b)

# Other analytical formulas

"""
  analytical_GeneralHunt_u(l, d_b, grad_pz, Ha, n, x)

General Hunt analytical formula (d_b = 0 for Shercliff).

# Arguments
- `l::Float64`: channel aspect ratio.
- `d_b::Float64`: Hartmann walls conductivity ratio.
- `grad_pz::Float64`: dimensionless (MHD version) presure gradient.
- `Ha::Float64`: Hartmann number.
- `n::Int`: number of sumands included in Fourier series.
- `x`: evaluation point normalized by the Hartmann characteristic lenght.
"""
function analytical_GeneralHunt_u(
  # General Hunt analytical formula (d_b = 0 for Shercliff)
  l::Real,          # channel aspect ratio
  d_b::Real,        # Hartmann walls conductivity ratio
  grad_pz::Real,    # Dimensionless (MHD version) presure gradient
  Ha::Real,         # Hartmann number
  n::Int,           # number of sumands included in Fourier series
  x)                # evaluation point normaliced by the Hartmann characteristic lenght

  V = 0.0
  V0 = 0.0
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

  VectorValue(0.0*u_z, 0.0*u_z, u_z)
end

"""
  surf_avg(model, mag, surf; restrict=x->1.0, degree=6)

Compute the average of `mag` over some surface `surf` defined as a tag on `model`.
`degree` sets the degree of the quadrature rule. `restrict` is function that multiplies
`mag` and the area integral. It can be used, e.g., to restrict the average to a certain
subdomain.

  `surf_avg = ∫(restrict*mag)d(model,surf) / ∫(restrict)d(model,surf)`.
"""
function surf_avg(model, mag, surf; restrict=x->1.0, degree=6)
  Γ = Boundary(model, tags=surf)
  dΓ = Measure(Γ, degree)
  mag_avg = sum(∫(restrict*mag)*dΓ)/sum(∫(restrict)*dΓ)

  return mag_avg
end


"""
  quad(f, x₀, x₁; n=100)

Simple quadrature of `f(x)` between `x₀` and `x₁` using `n` divisions.
"""
function quad(f, x₀, x₁; n=100)
  Δx = (x₁ - x₀)/n
  q = sum(map(f, x₀ .+ collect(1:n) .* Δx).*Δx)

  return q
end
