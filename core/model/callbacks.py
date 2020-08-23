from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import datetime
import os


class Callbacks():
    def __init__(self, checkpoint_dir, tensorboard_dir, csv_dir):
        """All args are Path objects
        """
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir
        self.csv_dir = csv_dir
        self.callbacks = []
        self()

    def __call__(self):
        self.callbacks.append(self.get_checkpoint_callback())
        self.callbacks.append(self.get_tensorboard_callback())
        self.callbacks.append(self.get_csv_callback())
        return self.callbacks

    def get_checkpoint_callback(self):
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "best_accuracy_weights_" + date + ".h5"
        filepath = self.checkpoint_dir / filename

        create_dir(self.checkpoint_dir)

        cb = ModelCheckpoint(
            filepath=filepath,
            monitor="val_accuracy",
            mode="max",
            save_weights_only=True,
            save_best_only=True
        )

        return cb

    def get_tensorboard_callback(self):
        create_dir(self.tensorboard_dir)

        cb = TensorBoard(
            log_dir=self.tensorboard_dir.as_posix(),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
            profile_batch=2
        )

        return cb

    def get_csv_callback(self):
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "training_" + date + ".csv"
        filepath = self.csv_dir / filename

        create_dir(self.csv_dir)

        cb = CSVLogger(
            filename=filepath,
            separator=',',
            append=False
        )

        return cb


def create_dir(dir):
    if not dir.exists():
        os.mkdir(dir)
