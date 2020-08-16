import tensorflow as tf


def parse_image(img_path: str,
                label_path: str) -> tuple:
    """Parse the image and the gt image ready for the training.

    Arguments:
    img_path (string): the absolute path to an image
    label_path(string): the absolute path to an image

    Returns:
    image (np array): shape=(224, 224, n_channels) the resized image
    label (np array): shape=(224, 224, 1) the resized gt image where each pixel
        is equal to the class id.
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])

    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    print(label.shape)
    label = tf.image.resize(label, size=[224, 224])
    label = tf.cast(label, dtype=tf.uint8)
    label = tf.one_hot(label, depth=35)
    label = tf.squeeze(label)
    print(label.shape)

    return image, label


@tf.function
def normalize(input_image: tf.Tensor,
              input_mask: tf.Tensor) -> tuple:

    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


@tf.function
def load_image_train(image: tf.Tensor,
                     label: tf.Tensor) -> tuple:

    image_size = (224, 224)
    input_image = tf.image.resize(image, image_size)
    input_mask = tf.image.resize(label, image_size)

    # TODO: augmentation
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


@tf.function
def load_image_test(image: tf.Tensor,
                    label: tf.Tensor) -> tuple:

    image_size = (224, 224)
    input_image = tf.image.resize(image, image_size)
    input_mask = tf.image.resize(label, image_size)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask
