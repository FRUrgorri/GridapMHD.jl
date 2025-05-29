#!/bin/bash

#SBATCH -N 1 
#SBATCH --ntasks-per-node=4
#SBATCH -t 02:00:00
#SBATCH --partition=xula3
#SBATCH --job-name=Test
#SBATCH --output=./data/%x-%j.out
#SBATCH --error=./data/%x-%j.err
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

ENV_FILE="/mnt/lustre/home/u6678/work/GridapCalc/env.sh"
source ${ENV_FILE}

echo "Model input list:"

nX="4"	
echo "nX:${nX}"
nY="4"	
echo "nY:${nY}"
nZ="4"	
echo "nZ:${nZ}"
Ha="10"	
echo "Ha:${Ha}"
Re="1"	
echo "Re:${Re}"
N="${Ha}^2/${Re}" 
b="1"	
echo "b/a:${b}"
L="1"	
echo "L/a:${L}"
JOB_NAME="${SLURM_JOB_NAME}_Ha${Ha}Re${Re}_${nX}x${nY}x${nZ}_s${SLURM_JOB_ID}"
DATAPATH="./data/${JOB_NAME}"

mkdir -p "$DATAPATH"

# For INFOG(1) = -20 error
#  -mat_mumps_icntl_14 50

JULIA_SCRIPT="
using GridapMHD
using SparseMatricesCSR
using GridapPETSc
using Gridap
#using NLsolve

#Me gustaría meter U_inlet y B en info, ver si se puede hacer desde dentro del driver o hay que hacerlo desde fuera

U_inlet((x,y,z))=VectorValue(0.0,0.0,GridapMHD.u_parabolic(${b})(x,y))
#B((x,y,z))=VectorValue(0.0,GridapMHD.B_Moreau(${z0})(z),0.0)
B((x,y,z))=VectorValue(0.0,1.0,0.0)

Model = GridapMHD.Meshers.channel_model((${nX},${nY},${nZ});
		Ha = ${Ha},
		b= ${b},
		L= ${L}
		)

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
      :u => U_inlet,
      :j => VectorValue(0.0,0.0,0.0),
      :p => 0.0,
      :φ => 0.0,
    ),
)

xh,Ω = SteadyState(;
  title = \"${JOB_NAME}\",
  path = \"${DATAPATH}/\",
  backend = :mpi,
  np = (1, 1, ${SLURM_NPROCS}),
  modelGen = Model,
  Ha = ${Ha},
  Re = nothing,
  N = ${N},
  Bfield = B,
  u_inlet = U_inlet,
#  kj = ${kj},
  mesh2vtk = false,
  solver = solver,
  convection = true,
)

GridapMHD.post_process(xh, Ω, B, \"${DATAPATH}/\", \"${JOB_NAME}\")

"

echo "$JULIA_SCRIPT" > ${DATAPATH}/${JOB_NAME}_run.jl
cat $ENV_FILE > ${DATAPATH}/${JOB_NAME}_env.sh

srun --mpi=pmix -n ${SLURM_NPROCS} julia --project=$GRIDAPMHD -O3 --check-bounds=no -e "${JULIA_SCRIPT}"

julia --project=$GRIDAPMHD -e "
using BSON
using Printf
try
  global data = BSON.load(\"${DATAPATH}/${JOB_NAME}.bson\")
catch Exc
  println(\"No BSON found\")
else
  for key in keys(data)
    @printf(\"%-20s %-20s\\n\", \"\$key:\", data[key])
  end
end
"

mv ./data/*.out ${DATAPATH} 
mv ./data/*.err ${DATAPATH}

duration=$SECONDS
STATUS=$?

echo "================================================================"
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
echo ""
echo "Return code: $STATUS"
echo "================================================================"
