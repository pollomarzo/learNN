import tensorflow as tf

print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

#better option
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())