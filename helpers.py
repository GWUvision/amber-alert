import tensorflow as tf

def get_image_shape():
	return (255, 255, 3)

def get_session():
    """ Construct a modified tf session.
    """
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
