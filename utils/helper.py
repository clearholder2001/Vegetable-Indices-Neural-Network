import inspect
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_train_history(train_history, train, validation, save_figure_path):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title = "Train History"
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, len(train_history.history[train]), step=1.0))
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_figure_path.joinpath("Train History.jpg"))
    plt.close(fig)


def output_init(cfg):
    # clean default output folder
    shutil.rmtree(cfg.OUTPUT_DEFAULT_PATH, ignore_errors=True)
    shutil.rmtree(cfg.TENSORBOARD_LOG_PATH, ignore_errors=True)
    cfg.OUTPUT_DEFAULT_PATH.mkdir(parents=True, exist_ok=True)
    cfg.SAVE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    cfg.SAVE_IMAGE_PATH.mkdir(parents=True, exist_ok=True)
    cfg.TENSORBOARD_LOG_PATH.mkdir(parents=True, exist_ok=True)
    print("Output folder is ready.")


def print_cfg(cfg):
    print("--------------------cfg--------------------")
    for setting in inspect.getmembers(cfg):
        if not setting[0].startswith('_') and not setting[0] == 'os':
            print(setting)
    print("--------------------end--------------------")
