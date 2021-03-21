import warnings
from time import time

from tensorflow.keras.callbacks import Callback


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', loss=0.0001, save_weight_path=None, verbose=0):
        super().__init__()
        self.monitor = monitor
        self.loss = loss
        self.verbose = verbose
        self.save_weight_path = save_weight_path

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires {0} available!".format(self.monitor), RuntimeWarning)

        if current < self.loss:
            if self.verbose > 0:
                print("Epoch {0}: early stopping THR".format(epoch))
            self.model.stop_training = True
        # save the weights in every epoch
        self.model.save_weights(self.save_weight_path.joinpath("weight_epoch_{0}.h5".format(epoch)))


class TimingCallback(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time() - self.epoch_time_start)
