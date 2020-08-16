from typing import List
import tensorflow as tf
from .dataset_helper import load_image_train, load_image_test, parse_image


class Dataset():
    def __init__(self,
                 img_paths: List['Path'],
                 label_paths: List['Path'],
                 batch_size: int,
                 buffer_size: int,
                 seed: int,
                 augment: bool):

        self.img_paths = img_paths
        self.label_paths = label_paths
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.seed = seed
        self.augment = augment
        self.len = len(img_paths)
        self.steps = self.len // self.batch_size

        self.ds = self.create_dataset()

    def create_dataset(self) -> tf.data.Dataset:
        ds = tf.data.Dataset.zip(
            (self.img_filemane_dataset(), self.label_filename_dataset())
        )
        ds = ds.shuffle(self.len, seed=self.seed)
        ds = ds.map(parse_image)

        if self.augment is True:
            ds = ds.map(load_image_train, num_parallel_calls=4)
            ds = ds.repeat()
        else:
            ds = ds.map(load_image_test, num_parallel_calls=4)

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(self.buffer_size)

        return ds

    def img_filemane_dataset(self) -> tf.data.Dataset:
        img_filenames = [fn.as_posix() for fn in self.img_paths]
        img_filename_ds = tf.data.Dataset.from_tensor_slices(img_filenames)
        return img_filename_ds

    def label_filename_dataset(self) -> tf.data.Dataset:
        label_filenames = [fn.as_posix() for fn in self.label_paths]
        label_filename_ds = tf.data.Dataset.from_tensor_slices(label_filenames)
        return label_filename_ds
