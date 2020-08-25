import sys

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from core.generator.generator_helper import my_generator
from core.model.callbacks import Callbacks

from pathlib import Path

from core.model.model import FCN8

import datetime


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: main.py dataset_path log_path")

    dataset_path = Path(sys.argv[1])
    pre_trained_dir = Path(sys.argv[1])
    log_path = Path(sys.argv[2])

    # Settings
    batch_size = 16
    n_classes = 1
    traffic_light_class_id = 19
    seed = 42
    width = 2048
    height = 1024

    print("[+] Creating data generators...")

    gt_path = dataset_path / "gtFine_trainvaltest/gtFine"
    img_path = dataset_path / "leftImg8bit_trainvaltest/leftImg8bit"

    # Create generators for training
    img_train_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.2, 1.0]
    ).flow_from_directory(
        img_path / "train",
        class_mode=None,
        batch_size=batch_size,
        target_size=(width, height),
        seed=seed)

    mask_train_generator = ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.2, 1.0]
    ).flow_from_directory(
        gt_path / "train",
        class_mode=None,
        batch_size=batch_size,
        target_size=(width, height),
        seed=seed)

    train_generator = my_generator(
        img_train_generator,
        mask_train_generator
    )

    n_train_img = len([x for x in (img_path/"train").rglob("*.png")])
    train_steps = n_train_img // batch_size
    print(f"Train steps: {train_steps}")

    # Create generators for validating
    img_val_generator = ImageDataGenerator(
        rescale=1./255
    ).flow_from_directory(
        img_path / "val",
        class_mode=None,
        batch_size=batch_size,
        target_size=(width, height),
        seed=seed)

    mask_val_generator = ImageDataGenerator().flow_from_directory(
        gt_path / "val",
        class_mode=None,
        batch_size=batch_size,
        target_size=(width, height),
        seed=seed
    )

    val_generator = my_generator(
        img_val_generator,
        mask_val_generator
    )

    n_val_img = len([x for x in (img_path/"val").rglob("*.png")])
    val_steps = n_val_img // batch_size
    print(f"Validation steps: {val_steps}")

    # Creating the FCN8 model
    print("[+] Creating the model...")
    VGG_weights_path = Path(
        pre_trained_dir / "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    # vgg = build_vgg(VGG_weights_path, width, height)
    vgg = tf.keras.applications.VGG16(
      include_top=False,
      weights="imagenet",
      input_tensor=None,
      input_shape=(width, height, 3)
    )
    model = FCN8(vgg, n_classes, width, height)
    # model.summary()

    print("[+] Compile the model...")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"])

    # Training the model
    print("[+] Training...")
    cb = Callbacks(
        checkpoint_dir=log_path / "checkpoint/",
        tensorboard_dir=log_path / "tensorboard_log/",
        csv_dir=log_path / "csv_log/"
    )

    results = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=20,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=cb.callbacks)

    # Saving the model
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = "last_epoch_weights"+date+".h5"
    filepath = log_path / filename
    model.save_weights(filepath)
