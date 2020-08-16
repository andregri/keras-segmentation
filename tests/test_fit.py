from .helper import TestSetup
from core.model.callbacks import Callbacks
from core.generator.dataset import Dataset


def test_callbacks():
    test_setup = TestSetup("data/", "data/")
    generators = test_setup.setup_generators(12, 12, 1)
    model = test_setup.setup_model()

    cb = Callbacks("data/chkpt/top_weights.h5", "data/logs/", "")

    model.fit(
        generators["train"],
        epochs=2,
        steps_per_epoch=generators["train"].__len__(),
        validation_data=generators["val"],
        validation_steps=generators["val"].__len__(),
        # This data generator has problems with callbacks based on
        # validation metrics.
        # TODO: https://github.com/keras-team/keras/issues/10472
        # Neither modelcheckpoint nor tensorboard works.
        callbacks=cb(checkpoint=False, tensorboard=False, csv=False),
        verbose=2
    )


def test_fit_datasets():
    test_setup = TestSetup("data/", "data/")
    img_paths, label_paths = test_setup.parse_dataset(12, 12, 0)

    train_ds = Dataset(
        img_paths["train"],
        label_paths["train"],
        batch_size=4,
        buffer_size=1000,
        seed=42,
        augment=True
    )

    val_ds = Dataset(
        img_paths["val"],
        label_paths["train"],
        batch_size=4,
        buffer_size=1000,
        seed=42,
        augment=False
    )

    model = test_setup.setup_model()

    model.fit(
        train_ds.ds,
        epochs=2,
        steps_per_epoch=train_ds.steps,
        validation_data=val_ds.ds,
        validation_steps=val_ds.steps,
        verbose=2
    )


def test_fit_datasets_callbacks():
    test_setup = TestSetup("data/", "data/")
    img_paths, label_paths = test_setup.parse_dataset(12, 12, 0)

    cb = Callbacks("data/chkpt/top_weights.h5", "data/logs/", "data/trainig.csv")

    train_ds = Dataset(
        img_paths["train"],
        label_paths["train"],
        batch_size=4,
        buffer_size=1000,
        seed=42,
        augment=True
    )

    val_ds = Dataset(
        img_paths["val"],
        label_paths["train"],
        batch_size=4,
        buffer_size=1000,
        seed=42,
        augment=False
    )

    model = test_setup.setup_model()

    model.fit(
        train_ds.ds,
        epochs=2,
        steps_per_epoch=train_ds.steps,
        validation_data=val_ds.ds,
        validation_steps=val_ds.steps,
        callbacks=cb(checkpoint=True, tensorboard=True, csv=True),
        verbose=2
    )
