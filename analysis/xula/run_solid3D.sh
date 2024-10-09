#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -t 24:00:00
#SBATCH --partition=xula3
#SBATCH --mail-user=guillermo.gomez@ciemat.es
#SBATCH --mail-type=END
#SBATCH --job-name=solid3D
#SBATCH --output=../data/%x-%j.out
#SBATCH --error=../data/%x-%j.err
#SBATCH --mem=0
###SBATCH --chdir=

SLURM_NPROCS=`expr $SLURM_JOB_NUM_NODES \* $SLURM_NTASKS_PER_NODE`

echo "================================================================"
hostname
echo "Using: ${SLURM_NPROCS} procs in ${SLURM_JOB_NUM_NODES} nodes"
echo "Date: `date +%Y-%m-%d\ %H:%M:%S`"
echo "================================================================"
echo ""

SECONDS=0

ENV_FILE="env_GGZ.sh"
source ${ENV_FILE}

nHa="50"
nSi="40"
nZ="25"
nsolid="5"
Ha="750"
N="${Ha}^2/1000"  # Re = Ha^2/N
cw_Ha="0.02377"
cw_s="0.02377"
tw_Ha="0.0222"
tw_s="0.0222"
b="1.0"
L="5"
JOB_NAME="${SLURM_JOB_NAME}_Ha${Ha}_${nHa}x${nSi}x${nZ}_s${SLURM_JOB_ID}"
DATAPATH="../data/${JOB_NAME}"

mkdir -p "$DATAPATH"

# For INFOG(1) = -20 error
#  -mat_mumps_icntl_14 50

JULIA_SCRIPT="
using GridapMHD: Solid
using SparseMatricesCSR
using GridapPETSc
using Gridap

solver = Dict(
    :solver => :petsc,
    :matrix_type    => SparseMatrixCSR{0,PetscScalar,PetscInt},
    :vector_type    => Vector{PetscScalar},
    :petsc_options  => \"-snes_monitor -ksp_error_if_not_converged true \\
                         -ksp_converged_reason -ksp_type preonly -pc_type lu \\
                         -pc_factor_mat_solver_type mumps -mat_mumps_icntl_28 1 \\
                         -mat_mumps_icntl_29 2 -mat_mumps_icntl_4 3 \\
                         -mat_mumps_cntl_1 0.001\",
    :niter          => 100,
    :rtol           => 1e-5,
    :initial_values => Dict(
      :u => VectorValue(0.0,0.0,1.0),
      :j => VectorValue(0.0,0.0,0.0),
      :p => 0.0,
      :φ => 0.0,
    ),
)

Solid(;
  title = \"${JOB_NAME}\",
  path = \"${DATAPATH}/\",
  backend = :mpi,
  np = (4, 4, 1),
  nl = (${nSi}, ${nHa}, ${nZ}),
  ns = (${nsolid}, ${nsolid}, 0),
  Ha = ${Ha},
  Re = nothing,
  N = ${N},
  B_var = :uniform,
  B_coef = nothing,
  dir_B = (0.0, 1.0, 0.0),
  cw_Ha = ${cw_Ha},
  cw_s = ${cw_s},
  b = ${b},
  L = ${L},
  tw_Ha = ${tw_Ha},
  tw_s = ${tw_s},
  inlet = :uniform,
  vtk = true,
  mesh2vtk = true,
  nsums = 100,
  stretch_fine = false,
  solver = solver,
)
"

echo "$JULIA_SCRIPT" > ${DATAPATH}/${JOB_NAME}_run.jl
cat $ENV_FILE > ${DATAPATH}/${JOB_NAME}_env.sh

srun --mpi=pmix julia --project=$GRIDAPMHD -O3 --check-bounds=no -e "${JULIA_SCRIPT}"

srun -n 1 julia -e "
using BSON
using Printf
try
  global data = BSON.load(\"${DATAPATH}/${JOB_NAME}_r1.bson\")
catch Exc
  println(\"No BSON found\")
else
  for key in keys(data)
    @printf(\"%-20s %-20s\\n\", \"\$key:\", data[key])
  end
end
"

duration=$SECONDS
STATUS=$?

echo "================================================================"
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
echo ""
echo "Return code: $STATUS"
echo "================================================================"
