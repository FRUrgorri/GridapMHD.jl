module load openmpi-4.1.5-gcc-13.1.0-nzcz542
module load gcc-13.1.0-gcc-8.5.0-ueufrru

export PETSCROOT=/mnt/lustre/home/u7482/software/petsc/petsc-3.15.4
export JULIA_PETSC_LIBRARY=/mnt/lustre/home/u7482/software/petsc/petsc-3.15.4/lib/libpetsc.so
export GMSHROOT=/mnt/lustre/home/u7482/software/gmsh/gmsh-4.13.1
export GRIDAPMHD=/mnt/lustre/home/u7482/software/GridapMHD/GridapMHD.jl.me

export OMPI_MCA_btl='^openib'
export OMPI_MCA_opal_warn_on_missing_libcuda=0
