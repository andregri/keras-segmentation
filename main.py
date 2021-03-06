import sys

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from core.generator.generator_helper import my_generator
from core.model.callbacks import Callbacks

from pathlib import Path

from core.model.model import FCN8, build_vgg

import datetime


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: main.py dataset_path log_path")

    dataset_path = Path(sys.argv[1])
    pre_trained_dir = Path(sys.argv[1])
    log_path = Path(sys.argv[2])

    print("[+] Creating data generators...")
    # Create generators
    gt_path = dataset_path / "gtFine_trainvaltest/gtFine"
    img_path = dataset_path / "leftImg8bit_trainvaltest/leftImg8bit"

    batch_size = 2

    img_train_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.2
    ).flow_from_directory(
        img_path / "train",
        class_mode=None,
        batch_size=batch_size,
        target_size=(224, 224),
        seed=42)

    mask_train_generator = ImageDataGenerator().flow_from_directory(
        gt_path / "train",
        class_mode=None,
        batch_size=batch_size,
        target_size=(224, 224),
        seed=42
    )

    train_generator = my_generator(img_train_generator, mask_train_generator)
    train_steps = len([x for x in (img_path/"train").rglob("*.png")]) // batch_size
    print(train_steps)

    img_val_generator = ImageDataGenerator(
        rescale=1./255
    ).flow_from_directory(
        img_path / "val",
        class_mode=None,
        batch_size=batch_size,
        target_size=(224, 224),
        seed=42)

    mask_val_generator = ImageDataGenerator().flow_from_directory(
        gt_path / "val",
        class_mode=None,
        batch_size=batch_size,
        target_size=(224, 224),
        seed=42
    )

    val_generator = my_generator(img_val_generator, mask_val_generator)
    val_steps = len([x for x in (img_path/"val").rglob("*.png")]) // batch_size
    print(val_steps)

    print("[+] Creating the model...")
    # Creating the FCN8 model
    VGG_weights_path = Path(
        pre_trained_dir / "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    vgg = build_vgg(VGG_weights_path, 224, 224)
    model = FCN8(vgg, 35, 224, 224)
    # model.summary()

    print("[+] Compile the model...")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    print("[+] Training...")
    # Training the model

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

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = "last_epoch_weights"+date+".h5"
    filepath = log_path / filename
    model.save_weights(filepath)
