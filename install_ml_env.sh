module purge
module load usc
module load anaconda3
module load cuda/11.1-1
module load nccl/2.8.3-1-cuda
module load pmix/2.2.2
conda create --name lfl_ml python=3.9
conda activate lfl_ml

conda install -c conda-forge tensorflow-gpu keras

MPI_BIN=$(which mpirun)
export MPI_DIR=${MPI_BIN%/*/*}
export LD_LIBRARY_PATH=${MPI_DIR}/lib:$LD_LIBRARY_PATH
export HOROVOD_NCCL_HOME=$NCCL_ROOT
export HOROVOD_WITH_TENSORFLOW=1
export HOROVOD_WITH_MPI=1
export HOROVOD_GPU_ALLREDUCE=NCCL

CC=mpicc CXX=mpicxx pip install --no-cache-dir horovod


