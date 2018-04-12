# coding=utf-8
__author__ = 'gu'

import numpy as np
from sklearn import preprocessing
np.seterr(divide='ignore', invalid='ignore')


def load_liner_data():
    """

    :return:
    """
    x_train = np.load("./fb_data/textmind_train_vec.npy")
    x_test = np.load("./fb_data/textmind_train_vec.npy")
    y_train = np.load("./fb_data/score_label.npy")
    y_test = np.load("./fb_data/score_label.npy")
    x_train = normall_all(x_train)
    x_test = normall_all(x_test)
    split_point = 50
    return x_train[split_point:], y_train[split_point:], x_test[:split_point], y_test[:split_point]


def load_classfy_data():
    """
    :return:
    """
    x_train = np.load("./fb_data/textmind_train_vec.npy")
    x_test = np.load("./fb_data/textmind_train_vec.npy")
    y_train = np.load("./fb_data/labels.npy")
    y_test = np.load("./fb_data/labels.npy")
    x_train = normall_all(x_train)
    x_test = normall_all(x_test)
    split_point = 50
    return x_train[split_point:], y_train[split_point:], x_test[:split_point], y_test[:split_point]


def normall(x_train, i):
    x1 = x_train[:, i]
    max = x1.max()
    min = x1.min()
    x1 = np.where(x1, (x1 - min) / (max - min), 0)
    return x1


def normall_all(x_train):
    """
    使用(x1 - min) / (max - min)归一化
    :return:
    """
    x_tr = []
    for i in range(102):
        x_traini = normall(x_train, i)
        x_tr.append(x_traini)
    x_tr = np.mat(x_tr)
    x_tr = np.transpose(x_tr)
    return x_tr

load_liner_data()