from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class Callbacks():
    def __init__(self, checkpoint_path, tensorboard_path):
        self.chkpt_path = checkpoint_path
        self.tb_path = tensorboard_path
        self.callbacks = []

    def __call__(self, checkpoint, tensorboard):
        if checkpoint is True:
            self.callbacks.append(self.get_checkpoint_callback())

        if tensorboard is True:
            self.callbacks.append(self.get_tensorboard_callback())

        return self.callbacks

    def get_checkpoint_callback(self):
        cb = ModelCheckpoint(
            filepath=self.chkpt_path,
            monitor="val_accuracy",
            mode="max",
            save_weights_only=True,
            save_best_only=True
        )

        return cb

    def get_tensorboard_callback(self):
        cb = TensorBoard(
            log_dir=self.tb_path,
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
            profile_batch=2
        )

        return cb