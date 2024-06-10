import os
import tensorflow as tf

print("CUDA_HOME:", os.environ.get('CUDA_HOME'))
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))
print("PATH:", os.environ.get('PATH'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

