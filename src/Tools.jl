"""
  arg_n_smallest_values(A::AbstractArray{T,N}, n::Integer)

Find the `n` smalles values in `A`.
"""
function indices_n_smallest(A::AbstractArray{T,N}, n::Integer) where {T,N}
  perm = sortperm(vec(A))
  ci = CartesianIndices(A)

  return ci[perm[1:n]]
end


"""
  wavg_interpolator(coords, vals, x; n=3)

Find the distance-weighted average of the `n` elements in `vals` closests to `x` according
to `coords`

`vals` is an array of values or vectors where each entry corresponds to the value of a
discretized function evaluated at the corresponding `coords` entry.  The returned value is
thediscretized exact value, if it exists, or the distance-weighted average of the `n`
closest elements.
"""
function wavg_interpolator(coords, vals, x; n=3)
  xarr = get_array(x)
  distances = norm.([ci .- xarr for ci in coords])
  if minimum(distances) == 0.0
    interp = vals[argmin(distances)]
  else
    ids = indices_n_smallest(distances, n)
    weights = inv.(distances[ids])/sum(inv.(distances[ids]))
    interp = sum(weights .* vals[ids])
  end

  return interp
end


"""
  wavg_interpolator(coords::AbstractMatrix, vals::AbstractMatrix, x; n=3)
"""
function wavg_interpolator(coords::AbstractMatrix, vals::AbstractMatrix, x; n=3)
  C = [coords[1:3,i] for i in 1:size(coords)[2]]
  V = [vals[1:3,i] for i in 1:size(vals)[2]]

  return wavg_interpolator(C, V, x; n=n)
end


"""
  get_coordinates(cp::CellPoint)

Return coordinates of CellPoint `cp` as a matrix.
"""
get_coordinates(cp::CellPoint) = _arr_to_matrix(cp.cell_phys_point)


"""
  get_coordinates(cp::GridapDistributed.DistributedCellPoint)
"""
function get_coordinates(cp::GridapDistributed.DistributedCellPoint)
  coord_arr = []
  map(local_views(cp)) do lv
    push!(coord_arr, lv.cell_phys_point)
  end

  return _arr_to_matrix(vcat(coord_arr...))
end


"""
  get_values(field, triangulation)

Return valus of `field` evaluated at `triangulation` as a matrix.
"""
function get_values(field, triangulation)
  x = get_cell_points(triangulation)
  f_trian = evaluate(field, x)

  return _arr_to_matrix(f_trian)
end


"""
  get_values(field, triangulation::GridapDistributed.DistributedTriangulation)
"""
function get_values(field, triangulation::GridapDistributed.DistributedTriangulation)
  x = get_cell_points(triangulation)
  f_trian = evaluate(field, x)
  vals_arr = []
  map(f_trian) do lv
    push!(vals_arr, lv)
  end

  return _arr_to_matrix(vcat(vals_arr...))
end


"""
  write_tabular(tofile, filename)

Write tabular data `tofile` (e.g., a Matrix) to disk under `filename`.
"""
function write_tabular(tofile, filename)
  @assert size(tofile)[2] == 6
  f = open(filename, "w")
  write(f, "x y z v1 v2 v3\n")
  for row in 1:size(tofile)[1]
    for col in tofile[row,:]
        write(f, "$(col) ")
    end
    write(f, "\n")
  end
  close(f)

  return nothing
end


"""
  write_tabular_append(tofile, filename)

Write tabular data `tofile` (e.g., a Matrix) to disk under `filename` in append mode,
without overwriting nor adding headers.
"""
function write_tabular_append(tofile, filename)
  f = open(filename, "a")
  for row in 1:size(tofile)[1]
    for col in tofile[row,:]
        write(f, "$(col) ")
    end
    write(f, "\n")
  end
  close(f)

  return nothing
end


"""
  safe_write_tabular(tofile, filename, ranks)

Write tabular data `tofile` (e.g., a Matrix) to disk under `filename` where data is
distributed along `ranks`.  IO operations are serialized to avoid ranks overwriting
each other.
"""
function safe_write_tabular(
  tofile, filename, model::GridapDistributed.DistributedModelOrTriangulation
)
  ranks = get_parts(model)
  if i_am_main(ranks)
    f = open(filename, "w")
    write(f, "x y z v1 v2 v3\n")
    close(f)
  end
  for i in 1:length(ranks)
    map(ranks) do rank
      if rank == i
        write_tabular_append(tofile, filename)
      end
    end
    PartitionedArrays.barrier(ranks)
  end

  return nothing
end


"""
  safe_write_tabular(tofile, filename, model)
"""
function safe_write_tabular(tofile, filename, model)
  write_tabular(tofile, filename)

  return nothing
end


"""
  todisk(model, field, filename; tag=nothing)

Save `field` on `model` boundary tagged `tag` to `filename`.
"""
function todisk(model, field, filename; tag=nothing)
  if isnothing(tag)
    trian = get_triangulation(model)
  else
    trian = BoundaryTriangulation(model, tags=[tag, ])
  end
  x = get_cell_points(trian)
  coords = get_coordinates(x)
  vals = get_values(field, trian)
  tabular = transpose(vcat(coords, vals))
  safe_write_tabular(tabular, filename, model)
end


"""
  read_tabular(filename; delimiter=" ", skip=1, n=0)

Read tabular data from `filename` and return it as a `Matrix`.

Silently drop lines with a number of elements different than `n` (if `n > 0`).
"""
function read_tabular(filename; delimiter=" ", skip=1, n=0)
  @assert isfile(filename)
  arr = []
  i = 1
  for line in eachline(filename)
    if i > skip
      newline = parse.(Float64, split(line, delimiter; keepempty=false))
      if n > 0 && length(newline) == n
        push!(arr, newline)
      end
    else
      i += 1
    end
  end

  return transpose(stack(arr))
end


function _arr_to_matrix(arr)
  point_arr = [[point...] for cell in arr for point in cell]
  if !isempty(point_arr)
    _arr = stack(point_arr)
  else
    _arr = Matrix{Float64}(undef, 3, 0)
  end

  return _arr
end
