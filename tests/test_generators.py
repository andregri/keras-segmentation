from .helper import TestSetup


def test_TestSetup():
    test_setup = TestSetup("data/", "data/")
    generators = test_setup.setup_generators(100, 10, 50)
    assert len(generators["train"].pairs) == 100
    assert len(generators["val"].pairs) == 10
    assert len(generators["test"].pairs) == 50
