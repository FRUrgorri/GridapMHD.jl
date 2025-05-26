module Meshers
  using DrWatson
  using PartitionedArrays, MPI
  using GridapDistributed
  using Gridap, GridapP4est, GridapSolvers

  using Gridap.Arrays, Gridap.Geometry, Gridap.ReferenceFEs, Gridap.Adaptivity

  include("tools.jl")
  export stretchMHD
  export ChangeDensity
  
  include("p4est.jl")
  
  export generate_refined_mesh
  export generate_mesh_hierarchy

  include("expansion_mesher.jl")

  export expansion_generate_base_mesh

  include("hunt_mesher.jl")
  
  export hunt_generate_base_mesh
  

  include("solid_mesher.jl")

  export solid_mesh_map, solid_add_tags!, solid_wall_BC
  
end
