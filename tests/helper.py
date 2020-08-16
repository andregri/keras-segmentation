from pathlib import Path

from core.generator.data_generator import DataGenerator
from core.generator.generator_helper import make_pair
from core.model.model import build_vgg, FCN8


class TestSetup():
    def __init__(self, dataset_dir, vgg_dir):
        """
        Arguments:
            dataset_path (string): path to the dataset dir.
            model_path (string): path to the pre-trained weights of vgg16.
        """
        self.dataset_path = Path(dataset_dir)
        self.pre_trained_path = Path(vgg_dir)

    def parse_dataset(self, n_train=None, n_val=None, n_test=None):
        gt_path = self.dataset_path / "gtFine_trainvaltest/gtFine"
        img_path = self.dataset_path / "leftImg8bit_trainvaltest/leftImg8bit"

        # Get the path of all images
        print("[+] Reading the dataset...")

        img_paths = {}
        label_paths = {}
        for key in ["train", "val", "test"]:
            img_paths[key] = [path for path in (img_path / key).rglob("*.png")]
            label_paths[key] = [path for path in (
                gt_path / key).rglob("*.png") if "labelIds" in path.name]

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

        if n_train is not None:
            img_paths["train"] = img_paths["train"][:n_train]
            label_paths["train"] = label_paths["train"][:n_train]

        if n_val is not None:
            img_paths["val"] = img_paths["val"][:n_val]
            label_paths["val"] = label_paths["val"][:n_val]

        if n_test is not None:
            img_paths["test"] = img_paths["test"][:n_test]
            label_paths["test"] = label_paths["test"][:n_test]

        return img_paths, label_paths

    def setup_generators(self, n_train=None, n_val=None, n_test=None):
        img_paths, label_paths = self.parse_dataset()

        # Create data pairs
        print("[+] Creating data pairs")

        data_pairs = {}
        for key in ["train", "val", "test"]:
            data_pairs[key] = make_pair(img_paths[key], label_paths[key])

        if n_train is not None:
            data_pairs["train"] = data_pairs["train"][:n_train]

        if n_val is not None:
            data_pairs["val"] = data_pairs["val"][:n_val]

        if n_test is not None:
            data_pairs["test"] = data_pairs["test"][:n_test]

        print(f"""
        TRAIN: {len(data_pairs['train'])}, {len(data_pairs['train'])}
        VAL:   {len(data_pairs['val'])}, {len(data_pairs['val'])}
        TEST:  {len(data_pairs['test'])}, {len(data_pairs['test'])}
        """)

        # Create generators
        print("[+] Creating data generators...")

        train_generator = DataGenerator(pairs=data_pairs["train"],
                                        batch_size=4,
                                        dim=(224, 224),
                                        shuffle=True)

        val_generator = DataGenerator(pairs=data_pairs["val"],
                                      batch_size=4,
                                      dim=(224, 224),
                                      shuffle=True)

        test_generator = DataGenerator(pairs=data_pairs["test"],
                                       batch_size=4,
                                       dim=(224, 224),
                                       shuffle=False)

        generators = {
            "train":    train_generator,
            "val":      val_generator,
            "test":     test_generator
        }

        return generators

    def setup_model(self):

        print("[+] Creating the model...")
        # Creating the FCN8 model
        vgg_weight_h5 = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
        VGG_weights_path = Path(self.pre_trained_path / vgg_weight_h5)
        vgg = build_vgg(VGG_weights_path, 224, 224)
        model = FCN8(vgg, 35, 224, 224)

        print("[+] Compile the model...")
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"])

        return model
