import datetime


def generate_slurm_script(shell_script_name, python_file_name):
    py_name = python_file_name.split(".")[0]
    
    # Define the SLURM script content
    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={py_name}
#SBATCH --output={py_name}%j.log
#SBATCH --error={py_name}%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=p100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
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
source $(conda info --base)/etc/profile.d/conda.sh
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
srun python {python_file_name}
"""

    # Write the SLURM script to a file
    with open(shell_script_name, 'w') as file:
        file.write(slurm_script_content)

# Example usage
py_name = 'train_cnn_v1'
generate_slurm_script(f'{py_name}.sh', f'{py_name}.py')
