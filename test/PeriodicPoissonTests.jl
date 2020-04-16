module PeriodicPoissonTests

include("../src/GridapMHD.jl")
using .GridapMHD
# using GridapMHD

using Gridap
using Test


model = CartesianDiscreteModel((0,1,0,1),(32,32),[1])
order = 3
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6])

trian = get_triangulation(model)
degree = 2*order
quad = CellQuadrature(trian,degree)

Vu = FESpace(
       reffe=:Lagrangian, order=order, valuetype=Float64,
       conformity=:H1, model=model, dirichlet_tags="dirichlet")

g(x) = 0
f(x) = x[1]*sin(5.0π*x[2])+exp(-((x[1]-0.5)^2+(x[2]-0.5)^2)/0.02)
U = TrialFESpace(Vu,g)

aa(u,v) = inner(∇(v),∇(u))
l(v) = v*f
t_Ω = AffineFETerm(aa,l,trian,quad)

op = AffineFEOperator(U,Vu,t_Ω)

uh = solve(op)

writevtk(trian,"results",nsubcells=order,cellfields=["u"=>uh])

@test 1==1 # check code gets here!


end #module