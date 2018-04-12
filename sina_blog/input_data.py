# coding=utf-8
__author__ = 'gu'

import numpy as np
from tensorflow.python.platform import gfile


def normall(x_train, i):
    x1 = x_train[:, i]
    max = x1.max()
    min = x1.min()
    x1 = np.where(x1, (x1 - min) / (max - min), 0)
    return x1


def load_x():
    """
    使用(x1 - min) / (max - min)归一化
    :return:
    """
    x_train = np.load("./data/textmind_train_vec.npy")
    x_test = np.load("./data/textmind_test_vec.npy")
    x_tr = []
    x_te = []
    for i in range(102):
        x_traini = normall(x_train, i)
        x_testi = normall(x_test, i)
        x_tr.append(x_traini)
        x_te.append(x_testi)
    x_te = np.mat(x_te)
    x_te = np.transpose(x_te)
    x_tr = np.mat(x_tr)
    x_tr = np.transpose(x_tr)
    return x_tr, x_te


def load_normall_x():
    """
    使用sklearn归一化
    :return:
    """
    from sklearn import preprocessing

    x_train = np.load("./data/textmind_train_vec.npy")
    x_test = np.load("./data/textmind_test_vec.npy")
    x_tr = preprocessing.scale(x_train)
    x_te = preprocessing.scale(x_test)
    print(x_tr)
    return x_tr, x_te


def usids(train_dir):
    userid_list = gfile.ListDirectory(train_dir)
    for uid in userid_list:
        print(uid)


def load_liner_y():
    label_train_path = "./data/label_train.txt"
    label_test_path = "./data/label_test.txt"
    y_train = get_labels(label_train_path)
    y_test = get_labels(label_test_path)
    y_train = np.mat(y_train)
    y_test = np.mat(y_test)
    print(y_train)
    print(y_test)
    return y_train, y_test


def get_labels(label_path):
    ys = []
    with gfile.Open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            words = line.split(' ')
            temp = []
            for value in words[1:]:
                temp.append(int(value))
            ys.append(temp)
    return ys
