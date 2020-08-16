from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger


class Callbacks():
    def __init__(self, checkpoint_path, tensorboard_path, csv_path):
        self.chkpt_path = checkpoint_path
        self.tb_path = tensorboard_path
        self.csv_path = csv_path
        self.callbacks = []

    def __call__(self, checkpoint, tensorboard, csv):
        if checkpoint is True:
            self.callbacks.append(self.get_checkpoint_callback())

        if tensorboard is True:
            self.callbacks.append(self.get_tensorboard_callback())

        if csv is True:
            self.callbacks.append(self.get_csv_callback())

        return self.callbacks

    def get_checkpoint_callback(self):
        cb = ModelCheckpoint(
            filepath=self.chkpt_path,
            monitor="val_accuracy",
            verbose=1,
            save_weights_only=True,
            save_best_only=True
        )

        return cb

    def get_tensorboard_callback(self):
        cb = TensorBoard(
            log_dir=self.tb_path,
            histogram_freq=1,
        )

        return cb

    def get_csv_callback(self):
        cb = CSVLogger(self.csv_path)
        return cb
