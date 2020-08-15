import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """Generates data for Keras
    See:  https://www.kaggle.com/mukulkr/camvid-segmentation-using-unet
          https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, pairs, batch_size, dim, shuffle):
        """Initialization

        Arguments:
          pair (list of tuples): each tuple is a pair that contains the path to
            an image and the path to a ground-truth image.
        """
        self.pairs = pairs
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        """Update indexes after each epoch if shuffle=True
        """
        # Assign an index (an integer number) to each pair.
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.indexes = np.arange(len(self.pairs))
            if self.shuffle is True:
                np.random.shuffle(self.indexes)

    def __len__(self):
        """Number of batches for epoch
        """
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data

        Arguments:
          index (int): index of the batch

        Returns:
          batch_x (numpy array): array of images
          batch_y (numpy array): array of ground-truth image
        """
        # Get a slice of `batch_size` ints from `indexes`
        batch_indexes = \
            self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = []
        batch_y = []

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            for i in batch_indexes:
                img = img_to_array(
                    load_img(self.pairs[i][0], target_size=self.dim)) / 255
                batch_x.append(img)

                gt_img = img_to_array(
                    load_img(self.pairs[i][1], target_size=self.dim))
                class_img = to_categorical(gt_img[:, :, 0], num_classes=35)
                batch_y.append(class_img)

        return np.array(batch_x), np.array(batch_y)
