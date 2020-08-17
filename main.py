import sys

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,
# ReduceLROnPlateau, CSVLogger, TensorBoard

from core.model.callbacks import Callbacks
from core.generator.dataset_helper import load_image_train, load_image_test, parse_image
from core.generator.generator_helper import make_pair

from pathlib import Path

from core.model.model import FCN8, build_vgg


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: main.py dataset_path")

    dataset_path = Path(sys.argv[1])
    pre_trained_dir = Path(sys.argv[1])

    print("[+] Reading the dataset...")
    # Get the path of all images
    print("\n".join((p.as_posix() for p in dataset_path.iterdir())))

    gt_path = dataset_path / "gtFine_trainvaltest" / "gtFine"
    img_path = dataset_path / "leftImg8bit_trainvaltest" / "leftImg8bit"

    img_paths = {}
    label_paths = {}
    for key in ["train", "val", "test"]:
        img_paths[key] = [path for path in (img_path / key).rglob("*.png")]
        # labelsId images have one channel and grey level represents the label
        # id.
        label_paths[key] = [
            path for path in (gt_path / key).rglob("*.png")
            if "labelIds" in path.name]

    print(f"""
    TRAIN: {len(img_paths['train'])}, {len(label_paths['train'])}
    VAL:   {len(img_paths['val'])}, {len(label_paths['val'])}
    TEST:  {len(img_paths['test'])}, {len(label_paths['test'])}
    """)

    for key in ["train", "val", "test"]:
        assert len(img_paths[key]) == len(label_paths[key]), \
            f"Number of {key} images and labels mismatch."

    for key in ["train", "val", "test"]:
        img_paths[key].sort(), label_paths[key].sort()

    data_pairs = {}
    for key in ["train", "val", "test"]:
        data_pairs[key] = make_pair(img_paths[key], label_paths[key])

    print(f"""
    TRAIN: {len(data_pairs['train'])}, {len(data_pairs['train'])}
    VAL:   {len(data_pairs['val'])}, {len(data_pairs['val'])}
    TEST:  {len(data_pairs['test'])}, {len(data_pairs['test'])}
    """)

    # Create datasets
    BATCH_SIZE = 32
    BUFFER_SIZE = 1000
    SEED = 42

    # TRAIN DATASET
    # Create a dataset containing the filenames of the images for training
    train_img_filenames = [fn.as_posix() for fn in img_paths["train"]]
    train_img_filename_ds = tf.data.Dataset.from_tensor_slices(
        train_img_filenames)

    # Create a dataset containing the filenames of the gt images for training
    train_label_filenames = [fn.as_posix() for fn in label_paths["train"]]
    train_label_filename_ds = tf.data.Dataset.from_tensor_slices(
        train_label_filenames)

    # Map the parse function over the dataset
    train_filename_ds = tf.data.Dataset.zip(
        (train_img_filename_ds, train_label_filename_ds))
    train_ds = train_filename_ds.shuffle(len(train_img_filenames), seed=SEED)
    train_ds = train_ds.map(parse_image)
    train_ds = train_ds.map(load_image_train, num_parallel_calls=4)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=1000)

    # VALIDATION DATASET
    # Create a dataset containing the filenames of the images for VALIDATION
    val_img_filenames = [fn.as_posix() for fn in img_paths["val"]]
    val_img_filename_ds = tf.data.Dataset.from_tensor_slices(val_img_filenames)

    # Create a dataset containing the filenames of the gt images for VALIDATION
    val_label_filenames = [fn.as_posix() for fn in label_paths["val"]]
    val_label_filename_ds = tf.data.Dataset.from_tensor_slices(
        val_label_filenames)

    # Map the parse function over the dataset
    val_filename_ds = tf.data.Dataset.zip(
        (val_img_filename_ds, val_label_filename_ds))
    val_ds = val_filename_ds.shuffle(len(val_img_filenames), seed=SEED)
    val_ds = val_ds.map(parse_image)
    val_ds = val_ds.map(load_image_test, num_parallel_calls=4)
    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=1000)

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
    cb = Callbacks("data/chkpt/top_weights.h5",
                   "data/logs/", "data/trainig.csv")

    EPOCHS = 1

    STEPS_PER_EPOCH = len(data_pairs['train']) // BATCH_SIZE
    VALIDATION_STEPS = len(data_pairs['val']) // BATCH_SIZE

    results = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=20,
        validation_data=val_ds,
        validation_steps=VALIDATION_STEPS,
        callbacks=cb(checkpoint=True, tensorboard=True, csv=True))
