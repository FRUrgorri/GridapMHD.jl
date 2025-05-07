module SolidTestsMPI

using GridapMHD: Solid
using Gridap

# Fully Developed
Solid(;
  backend=:mpi,
  np=(2,2),
  cw_Ha = 0.01,
  cw_s = 0.01,
  tw_Ha = 0.1,
  tw_s = 0.1,
)

# 3D, petsc
Solid(;
  backend=:mpi,
  np=(1,1,4),
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

end
