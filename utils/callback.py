from time import time

from tensorflow.keras.callbacks import Callback


class TimingCallback(Callback):
    def __init__(self, epochs):
        self.epochs = epochs

    def on_train_begin(self, logs=None):
        self.times = [None] * self.epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time()

    def on_epoch_end(self, epoch, logs=None):
        self.end = time()
        self.times[epoch] = self.end - self.start

    def on_train_end(self, logs=None):
        self.times = [t for t in self.times if t != None]


class SaveWeightCallback(Callback):
    def __init__(self, save_weight_path):
        super().__init__()
        self.save_flag = False
        self.save_weight_path = save_weight_path

    def on_epoch_end(self, epoch, logs=None):
        self.save_flag = False
        if epoch < 50:
            self.save_flag = True
        elif epoch < 100 and (epoch+1) % 2 == 0:
            self.save_flag = True
        elif epoch >= 100 and (epoch+1) % 5 == 0:
            self.save_flag = True

        if self.save_flag:
            self.model.save_weights(self.save_weight_path.joinpath("weight_epoch_{0}.h5".format(epoch)))
