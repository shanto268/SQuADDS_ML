#!/bin/bash
#SBATCH --job-name=arch_opt
#SBATCH --output=arch_opt_%j.log
#SBATCH --error=arch_opt_%j.err
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

# Run the hyperparameter and architecture optimization script
srun python optimize_dnn_new_arch.py
