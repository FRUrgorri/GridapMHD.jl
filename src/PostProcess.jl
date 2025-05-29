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
function post_process(xh, Ω, B, path, title)

  uh, ph, jh, φh = xh
  div_jh = ∇·jh
  div_uh = ∇·uh
  grad_p = ∇·ph
  
  cellfields=[
    "uh"=>uh,
    "ph"=>ph,
    "jh"=>jh,
    "phi"=>φh,
    "div_uh"=>div_uh,
    "div_jh"=>div_jh,
    "grad_p"=>grad_p,
    "B" => CellField(B, Ω)
  ]
  
   writevtk(Ω, joinpath(path, title), order=2, cellfields=cellfields)
   nothing
end
