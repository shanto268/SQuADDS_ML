#!/bin/bash
#SBATCH --job-name=squadds_data_process
#SBATCH --output=squadds_output_%j.log
#SBATCH --error=squadds_error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=23:59:59
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shanto@usc.edu
#SBATCH --chdir=/project/shaas_31/shanto/summer24/SQuADDS

# environment variables set
export QISKIT_METAL_HEADLESS=1

# Load the necessary modules
module load gcc/11.3.0
module load python/3.10.14
module load conda

# Activate the virtual environment if needed
conda activate squadds-env

# Run the Python script
srun python gen_par_data.py