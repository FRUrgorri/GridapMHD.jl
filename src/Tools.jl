"""
  arg_n_smallest_values(A::AbstractArray{T,N}, n::Integer)

Find the `n` smalles values in `A`.
"""
function indices_n_smallest(A::AbstractArray{T,N}, n::Integer) where {T,N}
  perm = sortperm(vec(A))
  ci = CartesianIndices(A)

  return ci[perm[1:n]]
end


function wavg_interpolator(coords::AbstractMatrix, vals::AbstractMatrix, x; n=8)
  C = [coords[1:3,i] for i in 1:size(coords)[2]]
  V = [vals[1:3,i] for i in 1:size(vals)[2]]

  return wavg_interpolator(C, V, x; n=n)
end


"""
  wavg_interpolator(coords, vals, x; n=8)

Find the distance-weighted average of the `n` elements in `vals` closests to `x` according
to `coords`

`vals` is an array of values or vectors where each entry corresponds to the value of a
discretized function evaluated at the corresponding `coords` entry.  The returned value is
thediscretized exact value, if it exists, or the distance-weighted average of the `n`
closest elements. 
"""
function wavg_interpolator(coords, vals, x; n=8)
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
  get_coordinates(cp::CellPoint)

Return coordinates of CellPoint `cp` as a matrix.
"""
get_coordinates(cp::CellPoint) = _arr_to_matrix(cp.cell_phys_point)

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
  get_boundary_value(model, tag, field)

Returns a CellField of `field` restricted to some `tag`ged boundary of `model`.
"""
function get_boundary_value(model, tags::AbstractArray, field)
  Γ = BoundaryTriangulation(model, tags=tags)
  x = get_cell_points(Γ)
  field_tag = evaluate(field, x)

  return field_tag, Γ
end


"""
  write_field(field, trian, file)

Save `field` evaluated at some triangulation `trian` alongside its evaluation coordinates
in some disk file named `file`.
"""
function write_field(field, trian, file)
  x = get_cell_points(trian)
  coords = _arr_to_matrix(x.cell_phys_points)
  vals = _arr_to_matrix(evaluate(field, x))
  tofile = transpose(vcat(coords, vals))
  write_tabular(tofile, file)

  return nothing
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
  read_tabular(filename; delimiter=" ", skip=1)

Read tabular data from `filename` and return it as a `Matrix`.
"""
function read_tabular(filename; delimiter=" ", skip=1)
  @assert isfile(filename)
  arr = []
  i = 1
  for line in eachline(filename)
    if i > skip
      push!(arr, parse.(Float64, split(line, delimiter; keepempty=false)))
    else
      i += 1
    end
  end

  return transpose(stack(arr))
end


_arr_to_matrix(arr) = stack([[point...] for cell in arr for point in cell])
