import inspect

import matplotlib.pyplot as plt


def show_train_history(train_history, train, validation):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title = "Train History"
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./fig/Train History.png')
    plt.close(fig)


def print_cfg(cfg):
    print("--------------------cfg--------------------")
    for setting in inspect.getmembers(cfg):
        if not setting[0].startswith('_') and not setting[0] == 'os':
            print(setting)
    print("--------------------end--------------------")
