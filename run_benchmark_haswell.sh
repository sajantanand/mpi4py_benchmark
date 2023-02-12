#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --time=00:30:00
#SBATCH --constraint=haswell
#SBATCH --output ./haswell.job_%J.out
#####SBATCH --output ./NPC_haswell.job_%J.out
##### #SBATCH --output ./haswell_pkl_intel-ompi.job_%J.out

#module load intel openmpi
module load blg_dmrg_haswell
module list
#conda activate /global/common/software/m3859/env

#module load anaconda3
# module load intel-mpi/gcc/2018.3/64
# conda activate mpi4py

# module load openmpi/gcc/3.0.3/64
# conda activate dedalus-slurm

srun -n 2 -c 64 --cpu-bind=cores python osu_bw.py
##srun -n 2 -c 64 --cpu-bind=cores python osu_bw_pkl.py

##srun -n 2 -c 64 --cpu-bind=cores python npc_benchmark.py

# srun python osu_bw.py

#conda activate mpi4py-slow
#mpirun python osu_bw.py
