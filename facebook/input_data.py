# coding=utf-8
__author__ = 'gu'

import numpy as np


def load_label():
    x_train = np.load("./fb_data/textmind_train_vec.npy")
    x_test = np.load("./fb_data/textmind_train_vec.npy")
    y_train = np.load("./fb_data/labes.npy")
    y_test = np.load("./fb_data/labes.npy")
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train, x_test, y_test


load_label()
