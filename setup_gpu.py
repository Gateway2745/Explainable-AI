import tensorflow as tf

MINIMUM_TF_VERSION = 1, 14, 0

BLACKLISTED_TF_VERSIONS = [
    (2, 0, 0),  # Has a number of memory leaks and issues with eager execution.
    (2, 0, 1),  # Has a number of memory leaks and issues with eager execution.
]

def tf_version():
    return tuple(map(int, tf.version.VERSION.split('-')[0].split('.')))

def tf_version_ok(minimum_tf_version=MINIMUM_TF_VERSION, blacklisted=BLACKLISTED_TF_VERSIONS):
    """ Check if the current Tensorflow version is higher than the minimum version.
    """
    return tf_version() >= minimum_tf_version and tf_version() not in blacklisted

def setup_gpu(gpu_id):
    if tf_version_ok((2, 0, 0)):
        if gpu_id == 'cpu' or gpu_id == -1:
            tf.config.experimental.set_visible_devices([], 'GPU')
            return

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only use the first GPU.
            try:
                # Currently, memory growth needs to be the same across GPUs.
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Use only the selcted gpu.
                tf.config.experimental.set_visible_devices(gpus[int(gpu_id)], 'GPU')
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized.
                print(e)

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    else:
        import os
        if gpu_id == 'cpu' or gpu_id == -1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
            return

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(config=config))