#!/bin/bash
#SBATCH --job-name=ml_training
#SBATCH --output=ml_training_%j.log
#SBATCH --error=ml_training_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40:1  # Request 1 K40 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shanto@usc.edu
#SBATCH --chdir=/project/shaas_31/shanto/summer24/SQuADDS_ML

# Load necessary modules
module purge
module load usc
module load anaconda3
module load cuda/11.1-1
module load nccl/2.8.3-1-cuda
module load pmix/2.2.2

# Check loaded modules
module list

# Activate Conda environment
source ~/miniconda3/bin/activate lfl_ml

# Set environment variables
MPI_BIN=$(which mpirun)
export MPI_DIR=${MPI_BIN%/*/*}
export LD_LIBRARY_PATH=${MPI_DIR}/lib:$LD_LIBRARY_PATH
export HOROVOD_NCCL_HOME=$NCCL_ROOT
export HOROVOD_WITH_TENSORFLOW=1
export HOROVOD_WITH_MPI=1
export HOROVOD_GPU_ALLREDUCE=NCCL
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH

# Create required directories
mkdir -p models figures

# Run the training script
srun python train_dnn_arch.py
