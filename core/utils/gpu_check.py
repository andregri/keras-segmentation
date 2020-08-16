import tensorflow as tf


def check_gpu():
    n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    print("Num GPUs Available: ", n_gpus)


check_gpu()
