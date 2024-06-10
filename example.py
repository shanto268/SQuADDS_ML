import tensorflow as tf

print(tf.sysconfig.get_build_info())

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# use this setting to find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

print(c)
