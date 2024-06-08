#!/bin/bash
#SBATCH --job-name=ml_test
#SBATCH --output=ml_test_data_%j.log
#SBATCH --error=ml_test.err  # Error log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shanto@usc.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G  # Increase memory allocation
#SBATCH --time=2:00:00

source /home1/shanto/miniconda3/etc/profile.d/conda.sh
conda activate lfl

# Set environment variable for threads
export OMP_NUM_THREADS=4

# Change to the directory where your scripts are located
cd /project/shaas_31/shanto/summer24/SQuADDS_ML

# Run the commands simultaneously in the background
python main.py --config config/neural_network_qubit_cavity.yml --no_hyper_opt &
python main.py --config config/lightgbm_qubit_cavity.yml --no_hyper_opt &
python main.py --config config/xgboost_qubit_cavity.yml --no_hyper_opt &
python main.py --config config/dnn_qubit_cavity.yml --no_hyper_opt &

# Wait for all background processes to complete
wait

echo "All processes have completed."
