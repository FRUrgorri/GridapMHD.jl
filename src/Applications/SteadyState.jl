"""
  SteadyState(; <keyword arguments>)

Driver capable of running 3D cases and fully developed approximation cases with solid
coupling in a rectangular geometry.

# Arguments
- To be defined
"""
function SteadyState(;
  backend = nothing,
  np = nothing,
  title = "MHD_SS",
#  nruns = 1,  #There is no interest a priory to repeate the same computation more than once (other than sudy scalability)
  path = ".",
  kwargs...
)

#  for ir in 1:nruns
#    _title = title*"_r$ir"
    if isa(backend,Nothing)
      @assert isa(np,Nothing)
      info, t, xh, Ω = _SteadyState(;title=title,path=path,kwargs...)
    else
      @assert backend ∈ [:sequential,:mpi]
      @assert !isa(np,Nothing)
      if backend === :sequential
        info, t, xh, Ω = with_debug() do distribute
          _SteadyState(;distribute=distribute,rank_partition=np,title=title,path=path,kwargs...)
        end
      else
        info,t, xh, Ω  = with_mpi() do distribute
          _SteadyState(;distribute=distribute,rank_partition=np,title=title,path=path,kwargs...)
        end
      end
 #   end

    info[:np] = np
    info[:backend] = backend
    info[:title] = title
    map_main(t.data) do data
      for (k,v) in data
        info[Symbol("time_$k")] = v.max
      end
      save(joinpath(path,"$title.bson"),info)
    end
  end

  return xh, Ω
end

function _SteadyState(;
  title = "MHD_SS",
  path = ".",
  distribute = nothing,
  rank_partition = nothing,
  modelGen = nothing,              
#  domain_tags = ("fluid",),           
  normalization = :mhd,                 
  Ha = 10.0,
  Re = 1.0,
  N = nothing,
  convection = true,
  Bfield = VectorValue(0.0,1.0,0.0),  
  u_inlet = VectorValue(0.0,0.0,1.0), 
  solve = true,
  solver = :julia,
  verbose = true,
  mesh2vtk = false,
  source = VectorValue(0.0, 0.0, 0.0),
#  μ = 0.0,
  ku = 2,
  kj = 1,
)

  info = Dict{Symbol,Any}()

  params = Dict{Symbol,Any}(
    :solve=>solve,
#    :res_assemble=>res_assemble,
#    :jac_assemble=>jac_assemble,
  )

  # Communicator
  if isa(distribute,Nothing)
    @assert isa(rank_partition,Nothing)
    rank_partition = Tuple(fill(1,3))     #Always 3D problems (even FD are computationally 3D)
    distribute = DebugArray
  end
  parts = distribute(LinearIndices((prod(rank_partition),)))
  
  # Timer
  t = PTimer(parts,verbose=verbose)
  params[:ptimer] = t
  tic!(t,barrier=true)

  # Solver
  if isa(solver,Symbol)
    solver = default_solver_params(Val(solver))
  end
  params[:solver] = solver
  
  #FE order
  params[:fespaces] = Dict{Symbol, Any}(
  :ku => ku,
  :kj => kj,
  )

  # Reduced quantities
  @assert normalization ∈ [:mhd,:cfd]
  if normalization == :mhd
    α = 1.0/N
    β = 1.0/Ha^2
    γ = 1.0
  else
    α = 1.0
    β = 1.0/Re
    γ = N
  end
  
  #Model from the input  function
  
  @assert !isa(modelGen,Nothing)
  model, tags_u, tags_j = modelGen(parts,rank_partition)
  
  params[:model] = model
  Ω = Interior(model)

  if mesh2vtk
    meshpath = joinpath(path, title*"_mesh")
    mkpath(meshpath)
    writevtk(model, meshpath)
  end
 
  #Prepare the input dictionary
  params[:fluid] = Dict{Symbol, Any}(
    :domain=>nothing, #For the moment, only fluid
    :α=>α,
    :β=>β,
    :γ=>γ,
    :f=>source,
    :B=>Bfield,
    :ζ=>0.0,
    :convection=>convection,
  )
   
 """ 
  if (tw_s > 0.0) || (tw_Ha > 0.0)
    params[:fluid][:domain] = "fluid"
    σ_Ω = σ_field(model, Ω, cw_Ha, cw_s, tw_Ha, tw_s)
    params[:solid] = Dict(:domain=>"solid", :σ=>σ_Ω)
  end
"""
  # Boundary conditions
    j_BC = Dict(
      :tags=>tags_j
    )
  if "inlet" in tags_u
    u_BC = Dict(
      :tags=>tags_u,
      :values=>[u_inlet, fill(VectorValue(0.0, 0.0, 0.0), length(tags_u)-1)...]
    )
  else
   u_BC = Dict(
      :tags=>tags_u
      )
  end
  
  params[:bcs] = Dict(
    :u=>u_BC,
    :j=>j_BC,
#    :thin_wall=>thinWall_params, #TBD
  )

"""
TBD: Allow a more general stabilization (at least a bit)
  # Stabilization method
  if μ > 0
    ĥ = b/nl[1]    See how the cell size is computed in GridapTritium
    params[:bcs][:stabilization] = Dict(:μ=>μ*ĥ, :domain=>"fluid")
  end
"""
  toc!(t,"pre_process")

  # Solve it
  if !uses_petsc(params[:solver])
    xh,fullparams,info = main(params;output=info)
  else
    petsc_options = params[:solver][:petsc_options]
    xh,fullparams,info = GridapPETSc.with(args=split(petsc_options)) do
      xh,fullparams,info = main(params;output=info)
      GridapPETSc.gridap_petsc_gc() # Destroy all PETSc objects
      return xh,fullparams,info
    end
  end

  t = fullparams[:ptimer]

  if verbose
    display(t)
  end


#This info is now much limited
  info[:ncells] = num_cells(model)
  info[:ndofs_u] = length(get_free_dof_values(xh[1]))
  info[:ndofs_p] = length(get_free_dof_values(xh[2]))
  info[:ndofs_j] = length(get_free_dof_values(xh[3]))
  info[:ndofs_φ] = length(get_free_dof_values(xh[4]))
  info[:ndofs] = length(get_free_dof_values(xh))
  info[:Re] = Re
  info[:Ha] = Ha
  info[:N] = N
#  info[:cw] = cw
  info[:convection] = convection
#  info[:μ] = μ

  return info, t, xh, Ω 
end

