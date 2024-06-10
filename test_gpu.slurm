#!/bin/bash
#SBATCH --job-name=ml_gpu_test
#SBATCH --output=gpu_test_%j.log
#SBATCH --error=gpu_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shanto@usc.edu
#SBATCH --chdir=/project/shaas_31/shanto/summer24/SQuADDS_ML

# Load necessary modules
module purge
module load usc
module load gcc/8.3.0
module load cuda/11.1-1
module load nccl/2.8.3-1-cuda
module load pmix/2.2.2

# Check loaded modules
module list

# Activate Conda environment
source ~/miniconda3/bin/activate lfl

# Set environment variables
MPI_BIN=$(which mpirun) ; export MPI_DIR=${MPI_BIN%/*/*}
export LD_LIBRARY_PATH={MPI_DIR}/lib:$LD_LIBRARY_PATH

# Check CUDA installation
nvcc --version

# Check GPU availability
nvidia-smi

# Run the test script
srun python test_gpu.py

