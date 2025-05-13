module SolidTestsSequential

using GridapMHD: Solid
using Gridap

# Fully Developed
Solid(;
  cw_Ha = 0.01,
  cw_s = 0.01,
  tw_Ha = 0.1,
  tw_s = 0.1,
)

# 3D, petsc
Solid(;
  nl = (10, 10, 4),
  ns = (5, 5, 0),
  Ha = 1.0,
  Re = nothing,
  N = 1.0,
  cw_Ha = 0.01,
  cw_s = 0.01,
  b = 1.0,
  L = 1.0,
  tw_Ha = 0.2,
  tw_s = 0.2,
  inlet = :uniform,
  mesh2vtk = true,
  solver = :petsc,
)

# Non uniform field
Blin_inc = GridapMHD.B_polynomial(0.5, 0.5)
Solid(;
  nl=(10,10,5),
  ns=(3,3,0),
  Re=nothing,
  Ha=10.0,
  N=1.0,
  L=1.0,
  inlet=:uniform,
  tw_Ha=0.1,
  tw_s=0.1,
  cw_Ha=0.1,
  cw_s=0.1,
  B_func=Blin_inc,
)

end
