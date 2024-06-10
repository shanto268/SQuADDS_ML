
MPI_BIN=$(which mpirun)
export MPI_DIR=${MPI_BIN%/*/*}
export LD_LIBRARY_PATH=${MPI_DIR}/lib:$LD_LIBRARY_PATH
export HOROVOD_NCCL_HOME=$NCCL_ROOT
export HOROVOD_WITH_TENSORFLOW=1
export HOROVOD_WITH_MPI=1
export HOROVOD_GPU_ALLREDUCE=NCCL

CC=mpicc CXX=mpicxx pip install --no-cache-dir horovod


