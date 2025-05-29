#Post processing functions for the SteadyState driver

"""
post_process(args)

Most basic postprocess function, it only generates the cell fields and write them in paraview

#Arguments

-`xh : Solution CellFields
-`B : External magnetic field imposed
-`Ω: Model interior
-`path: path for the output vtk file
-`title: title for the output vtk file

"""
function post_process(xh, B::VectorValue{3, Float64}, Ω, path, title)

  uh, ph, jh, φh = xh
  div_jh = ∇·jh
  div_uh = ∇·uh
  Grad_p = ∇·ph
  
  cellfields=[
    "uh"=>uh,
    "ph"=>ph,
    "jh"=>jh,
    "phi"=>φh,
    "div_uh"=>div_uh,
    "div_jh"=>div_jh,
    "grad_p"=>Grad_p,
    "B" => CellField(Bfield, Ω)
  ]
  
   writevtk(Ω, joinpath(path, title), order=max(ku,kj), cellfields=cellfields)
   nothing
end
