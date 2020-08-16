import sys

from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,
# ReduceLROnPlateau, CSVLogger, TensorBoard

from core.generator.dataset import Dataset

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

    # Create datasets
    train_ds = Dataset(
        img_paths=img_paths["train"],
        label_paths=label_paths["train"],
        batch_size=32,
        buffer_size=1000,
        seed=42,
        augment=True
    )

    val_ds = Dataset(
        img_paths=img_paths["val"],
        label_paths=label_paths["val"],
        batch_size=16,
        buffer_size=1000,
        seed=42,
        augment=False
    )

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
    results = model.fit(
        train_ds.ds,
        steps_per_epoch=train_ds.steps,
        epochs=20,
        validation_data=val_ds.ds,
        validation_steps=val_ds.steps)

    print("[+] Evaluate the model...")
    # Create the test generator and evaluate the model
    test_ds = Dataset(
        img_paths=img_paths["test"],
        label_paths=label_paths["test"],
        batch_size=16,
        buffer_size=1000,
        seed=42,
        augment=False
    )

    res = model.evaluate(
        test_ds.ds,
        steps=test_ds.steps,
        verbose=2)
