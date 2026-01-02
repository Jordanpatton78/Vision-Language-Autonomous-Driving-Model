#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=h200:1
#SBATCH --mem-per-cpu=32G   # memory per CPU core
#SBATCH -J "Distillation"   # job name
#SBATCH --output=../results/distillation_output.txt # change name of output file


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load python/3.11
source /home/patton78/cs674/advanced-deep-learning/dl_venv/bin/activate
python3 distillation.py
