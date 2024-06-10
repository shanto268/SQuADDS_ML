#!/bin/bash
#SBATCH --job-name=ml_training
#SBATCH --output=ml_training_%j.log
#SBATCH --error=ml_training_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=p100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shanto@usc.edu
#SBATCH --chdir=/project/shaas_31/shanto/summer24/SQuADDS_ML

# Load necessary modules
module purge
module load conda
module load gcc/11.3.0
module load python/3.11.3
module load cuda/12.0.0

# Check loaded modules
module list

# Activate Conda environment
conda activate lfl_ml

# Set environment variables
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH

# Check CUDA installation
nvcc --version

# Check GPU availability
nvidia-smi

# Create required directories
mkdir -p models figures

# Run the training script
srun python train_dnn_arch.py
