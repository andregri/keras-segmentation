import sys
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from core.generator.generator_helper import my_generator
from core.model.model import FCN8, build_vgg
from matplotlib import pyplot as plt
from core.preprocessing.cityscape.helpers import color_gt_image, IoU
from core.preprocessing.cityscape.cityscapes_labels import labels


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: main.py dataset_path weights_path")

    dataset_path = Path(sys.argv[1])
    pre_trained_dir = Path(sys.argv[1])
    weight_path = Path(sys.argv[2])

    print("[+] Create the model and load the weights")
    # Crete the model
    VGG_weights_path = Path(
        pre_trained_dir / "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    vgg = build_vgg(VGG_weights_path, 224, 224)
    model = FCN8(vgg, 35, 224, 224)
    model.load_weights(weight_path.as_posix())

    print("[+] Creating data generators...")
    # Create the generators
    gt_path = dataset_path / "gtFine_trainvaltest/gtFine"
    img_path = dataset_path / "leftImg8bit_trainvaltest/leftImg8bit"

    img_test_generator = ImageDataGenerator(
        rescale=1./255
    ).flow_from_directory(
        img_path / "val",
        class_mode=None,
        batch_size=16,
        target_size=(224, 224),
        seed=42)

    mask_test_generator = ImageDataGenerator().flow_from_directory(
        gt_path / "val",
        class_mode=None,
        batch_size=16,
        target_size=(224, 224),
        seed=42
    )

    test_generator = my_generator(img_test_generator, mask_test_generator)
    test_steps = len([x for x in (img_path/"train").rglob("*.png")]) // 16
    print(test_steps)

    print("[+] Predict some images...")
    (img, mask) = next(test_generator)  # batch of 16 images
    labels_mask = np.argmax(mask, axis=-1)
    onehot_predictions = model.predict(img)
    labels_prediction = np.argmax(onehot_predictions, axis=-1)

    id2name = {label.id: label.name for label in labels}
    id2color = {label.id: label.color for label in labels}
    for i in [4]:
        IoU(labels_mask[i], labels_prediction[i], id2name)

        _, axs = plt.subplots(1, 3)
        axs = axs.flatten()

        axs[0].imshow(img[i])
        axs[1].imshow(color_gt_image(labels_mask[i], id2color)/255.0)
        axs[2].imshow(color_gt_image(labels_prediction[i], id2color)/255.0)

        plt.show()
