# coding=utf-8
"""Functions for reading character data."""
import os

import numpy as np
from crawl_textmind_data import input_textmind_data


def load_data_label(base_model_dir):
    train_vec_filename = os.path.join(base_model_dir, "../svm_baseline_baobao/tfidf_train_vec_tfidf.npy")
    train_label_filename = os.path.join(base_model_dir, '../Emotion_Lexicon/emotion_train_label.npy')
    test_vec_filename = os.path.join(base_model_dir, '../svm_baseline_baobao/tfidf_test_vec_tfidf.npy')
    test_label_filename = os.path.join(base_model_dir, '../Emotion_Lexicon/emotion_test_label.npy')

    X_train = np.load(train_vec_filename)
    print('X_train', X_train.shape)
    Y_train = np.load(train_label_filename)
    print('Y_train', Y_train.shape)
    X_test = np.load(test_vec_filename)
    print('X_test', X_test.shape)
    Y_test = np.load(test_label_filename)
    print('Y_test', Y_test.shape)
    return X_train, Y_train, X_test, Y_test

def load_data_label1(base_model_dir):
    train_vec_filename = os.path.join(base_model_dir, "../svm_baseline_baobao/doc2vec_train_vec_dbow.npy")
    train_label_filename = os.path.join(base_model_dir, '../Emotion_Lexicon/emotion_train_label.npy')
    test_vec_filename = os.path.join(base_model_dir, '../svm_baseline_baobao/doc2vec_test_vec_dbow.npy')
    test_label_filename = os.path.join(base_model_dir, '../Emotion_Lexicon/emotion_test_label.npy')

    X_train = np.load(train_vec_filename)
    print('X_train', X_train.shape)
    Y_train = np.load(train_label_filename)
    print('Y_train', Y_train.shape)
    X_test = np.load(test_vec_filename)
    print('X_test', X_test.shape)
    Y_test = np.load(test_label_filename)
    print('Y_test', Y_test.shape)
    return X_train, Y_train, X_test, Y_test

def load_data_label_combine(X_train, X_test, X1_train, X1_test):
    """
    combine two arr into one
    :return:
    """
    X_train_all = np.hstack((X_train, X1_train))
    X_test_all = np.hstack((X_test, X1_test))
    return X_train_all, X_test_all


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data_label('')
    print(X_test)
    print(Y_test)
    X_train, Y_train, X1_test, Y1_test = input_textmind_data.load_textmind_data_label('../crawl_textmind_data')
    print(X1_test)
    print(Y1_test)
    X_test_all = np.hstack((X_test, X1_test))
    print(X_test_all.shape)
