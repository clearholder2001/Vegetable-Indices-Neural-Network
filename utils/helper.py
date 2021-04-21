import inspect
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score


def plot_train_history(train_history, train, validation, save_figure_path):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title = "Train History"
    plt.ylabel(train)
    plt.xlabel('Epoch')
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
        if not setting[0].startswith('_') and not setting[0] == 'Path':
            print(setting)
    print("--------------------end--------------------")


def calculate_statistics(test_Y, predict):
    assert test_Y.shape == predict.shape, 'Dimension inconsistent: test_Y, predict'
    rmse = np.sqrt(np.mean(np.square(test_Y - predict)))
    r2 = r2_score(test_Y.reshape(-1), predict.reshape(-1))
    correlation = stats.pearsonr(test_Y.reshape(-1), predict.reshape(-1))
    print("Final RMSE: {0:.4f}".format(rmse))
    print("Final R Square: {0:.4f}".format(r2))
    print("Final Correlation: {0:.4f}".format(correlation[0]))
