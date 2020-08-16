from .helper import TestSetup


def test_evaluate():
    test_setup = TestSetup("data/", "data/")
    generators = test_setup.setup_generators(1, 1, 50)
    model = test_setup.setup_model()
    model.evaluate(
        generators["test"],
        steps=generators["test"].__len__(),
        verbose=2
    )
