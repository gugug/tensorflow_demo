# coding=utf-8
"""Functions for reading character data."""
import os

import numpy as np


def load_data_label(base_model_dir):
    train_vec_filename = os.path.join(base_model_dir, "doc2vec_train_vec_dm.npy")
    train_label_filename = os.path.join(base_model_dir, 'doc2vec_train_label_dm.npy')
    test_vec_filename = os.path.join(base_model_dir, 'doc2vec_test_vec_dm.npy')
    test_label_filename = os.path.join(base_model_dir, 'doc2vec_test_label_dm.npy')

    X_train = np.load(train_vec_filename)
    print('X_train', X_train.shape)
    Y_train = np.load(train_label_filename)
    print('Y_train', Y_train.shape)
    X_test = np.load(test_vec_filename)
    print('X_test', X_test.shape)
    Y_test = np.load(test_label_filename)
    print('Y_test', Y_test.shape)
    return X_train, Y_train, X_test, Y_test


# if __name__ == '__main__':
    # X_train, Y_train, X_test, Y_test = load_data_label('')
    # print(X_test)
    # print(Y_test[:, 0])  # 取第i列
    # print np.where(X_test > 0, 1, 0)  # 对每一个元素判断 大于0为1,否则为0
