"""
A test process designed to run long enough to get a chance to actually see it
while running on the GPU, say by checking via nvidia-smi command.
"""

import tensorflow as tf
import time

# Ensure TensorFlow is using GPU
tf.debugging.set_log_device_placement(True)

# Define large matrices
size = 10000
matrix1 = tf.random.normal([size, size], dtype=tf.float32)
matrix2 = tf.random.normal([size, size], dtype=tf.float32)

# Repeat matrix multiplication 100 times
start_time = time.time()
for _ in range(100):
    result = tf.matmul(matrix1, matrix2)
    tf.reduce_sum(result).numpy()  # This forces the computation to happen

duration = time.time() - start_time
print(f"Repeated matrix multiplication completed in {duration:.2f} seconds")

