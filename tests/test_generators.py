from .helper import TestSetup
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import numpy as np


def test_TestSetup():
    test_setup = TestSetup("data/", "data/")
    generators = test_setup.setup_generators(100, 10, 50)
    assert len(generators["train"].pairs) == 100
    assert len(generators["val"].pairs) == 10
    assert len(generators["test"].pairs) == 50


def my_generator(img_gen, mask_gen):
    gen = zip(img_gen, mask_gen)
    for (img, mask) in gen:
        mask = mask[:, :, :, 0]
        mask = to_categorical(mask, num_classes=35)
        print(mask.shape)
        yield (img, mask)


def test_keras_generators():
    gt_path = "data/p/gtFine_trainvaltest/gtFine/train"
    img_path = "data/p/leftImg8bit_trainvaltest/leftImg8bit/train"

    img_generator = ImageDataGenerator(
        rotation_range=10
    ).flow_from_directory(
        img_path,
        class_mode=None,
        batch_size=4,
        target_size=(224, 224),
        seed=42)

    mask_generator = ImageDataGenerator(
        rotation_range=10
    ).flow_from_directory(
        gt_path,
        class_mode=None,
        batch_size=4,
        target_size=(224, 224),
        seed=42
    )

    my_gen = my_generator(img_generator, mask_generator)
    img, mask = next(my_gen)
    assert img.shape == (4, 224, 224, 3), "Error img"
    assert mask.shape == (4, 224, 224, 35), "Error mask"

    _, axs = plt.subplots(1, 2)
    axs = axs.flatten()
    axs[0].imshow(img[0]/255.0)
    m = np.argmax(mask[0].astype(int), axis=-1)
    print(m.shape)
    axs[1].imshow(m)
    plt.show()
