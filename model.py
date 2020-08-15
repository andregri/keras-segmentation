from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose,\
    Activation, Add


def build_vgg(weights_path, input_width=224, input_height=224):
    # input_height and width must be devisible by 32 because maxpooling with
    # filter size = (2,2) is operated 5 times, which makes the input_height and
    # width 2^5 = 32 times smaller

    assert input_width % 32 == 0
    assert input_height % 32 == 0
    IMAGE_ORDERING = "channels_last"

    # VGG model
    vgg = keras.models.Sequential()

    # Block1
    vgg.add(Conv2D(64, (3, 3), activation="relu", padding="same",
                   name="block1_conv1", data_format=IMAGE_ORDERING,
                   input_shape=(input_width, input_height, 3)))
    vgg.add(Conv2D(64, (3, 3), activation="relu", padding="same",
                   name="block1_conv2", data_format=IMAGE_ORDERING))
    vgg.add(MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool",
                         data_format=IMAGE_ORDERING))

    # Block2
    vgg.add(Conv2D(128, (3, 3), activation="relu", padding="same",
                   name="block2_conv1", data_format=IMAGE_ORDERING,
                   input_shape=(input_width, input_height, 3)))
    vgg.add(Conv2D(128, (3, 3), activation="relu", padding="same",
                   name="block2_conv2", data_format=IMAGE_ORDERING))
    vgg.add(MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool",
                         data_format=IMAGE_ORDERING))

    # Block3
    vgg.add(Conv2D(256, (3, 3), activation="relu", padding="same",
                   name="block3_conv1", data_format=IMAGE_ORDERING,
                   input_shape=(input_width, input_height, 3)))
    vgg.add(Conv2D(256, (3, 3), activation="relu", padding="same",
                   name="block3_conv2", data_format=IMAGE_ORDERING))
    vgg.add(Conv2D(256, (3, 3), activation="relu", padding="same",
                   name="block3_conv3", data_format=IMAGE_ORDERING))
    vgg.add(MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool",
                         data_format=IMAGE_ORDERING))

    # Block4
    vgg.add(Conv2D(512, (3, 3), activation="relu", padding="same",
                   name="block4_conv1", data_format=IMAGE_ORDERING,
                   input_shape=(input_width, input_height, 3)))
    vgg.add(Conv2D(512, (3, 3), activation="relu", padding="same",
                   name="block4_conv2", data_format=IMAGE_ORDERING))
    vgg.add(Conv2D(512, (3, 3), activation="relu", padding="same",
                   name="block4_conv3", data_format=IMAGE_ORDERING))
    vgg.add(MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool",
                         data_format=IMAGE_ORDERING))

    # Block5
    vgg.add(Conv2D(512, (3, 3), activation="relu", padding="same",
                   name="block5_conv1", data_format=IMAGE_ORDERING,
                   input_shape=(input_width, input_height, 3)))
    vgg.add(Conv2D(512, (3, 3), activation="relu", padding="same",
                   name="block5_conv2", data_format=IMAGE_ORDERING))
    vgg.add(Conv2D(512, (3, 3), activation="relu", padding="same",
                   name="block5_conv3", data_format=IMAGE_ORDERING))
    vgg.add(MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool",
                         data_format=IMAGE_ORDERING))

    # Dense layers of the original VGG16 model are discarded

    vgg.load_weights(weights_path.as_posix())

    return vgg


def FCN8(vgg, num_classes, input_width=224, input_height=224):
    assert input_width % 32 == 0
    assert input_height % 32 == 0
    IMAGE_ORDERING = "channels_last"

    # Deconvolution layers of the FCN8
    pool5 = vgg.get_layer("block5_pool").output
    fctoconv_1 = (Conv2D(4096, (7, 7), activation="relu", padding="same",
                         name="fctoconv_1",
                         data_format=IMAGE_ORDERING))(pool5)
    fctoconv_2 = (Conv2D(4096, (1, 1), activation="relu", padding="same",
                         name="fctoconv_2",
                         data_format=IMAGE_ORDERING))(fctoconv_1)

    # 2 times up-sampling
    deconv1 = Conv2DTranspose(num_classes, kernel_size=(2, 2), strides=(2, 2),
                              use_bias=False, data_format=IMAGE_ORDERING)
    (fctoconv_2)

    # 2 times up-sampling
    pool4 = vgg.get_layer("block4_pool").output
    predict2 = (Conv2D(num_classes, (1, 1), activation="relu", padding="same",
                       name="predict2", data_format=IMAGE_ORDERING))(pool4)
    sum1 = Add(name="add1")([predict2, deconv1])
    deconv2 = Conv2DTranspose(num_classes, (2, 2), strides=(2, 2),
                              use_bias=False,
                              data_format=IMAGE_ORDERING)(sum1)

    # 8 times up-sampling
    pool3 = vgg.get_layer("block3_pool").output
    predict3 = (Conv2D(num_classes, (1, 1), activation="relu", padding="same",
                       name="predict3", data_format=IMAGE_ORDERING))(pool3)
    sum2 = Add(name="add2")([deconv2, predict3])
    deconv3 = (Conv2DTranspose(num_classes, kernel_size=(8, 8), strides=(
        8, 8), use_bias=False, data_format=IMAGE_ORDERING))(sum2)

    output = (Activation("softmax"))(deconv3)

    model = keras.Model(vgg.input, output)

    return model
