#!/bin/bash
#SBATCH --job-name=v1_dnn_opt
#SBATCH --output=v1_dnn_opt_data_%j.log
#SBATCH --error=v1_dnn_opt.err  # Error log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shanto@usc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G  # Increase memory allocation
#SBATCH --time=23:00:00

source /home1/shanto/miniconda3/etc/profile.d/conda.sh
conda activate lfl

# Change to the directory where your scripts are located
cd /project/shaas_31/shanto/summer24/SQuADDS_ML 

# Run the commands simultaneously in the background
srun python optimize_dnn_arch_v1.py

