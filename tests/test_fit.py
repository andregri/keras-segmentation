from .helper import TestSetup
from core.model.callbacks import Callbacks


def test_callbacks():
    test_setup = TestSetup("data/", "data/")
    generators = test_setup.setup_generators(12, 12, 1)
    model = test_setup.setup_model()

    cb = Callbacks("data/chkpt/top_weights.h5", "data/logs/")

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
        callbacks=cb(checkpoint=False, tensorboard=False),
        verbose=2
    )
