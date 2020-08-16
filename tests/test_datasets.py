from core.generator.dataset import Dataset
from .helper import TestSetup


def test_dataset():
    test_setup = TestSetup("data/", "data/")
    img_paths, label_paths = test_setup.parse_dataset(12, 0, 0)

    train_ds = Dataset(
        img_paths["train"],
        label_paths["train"],
        batch_size=4,
        buffer_size=1000,
        seed=42,
        augment=True
    )

    assert train_ds.steps == 12 // 4
