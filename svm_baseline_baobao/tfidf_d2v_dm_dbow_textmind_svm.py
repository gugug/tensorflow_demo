# -*- coding: UTF-8 -*-
""" svm tfidf_d2v_dm_dbow_textmind for character"""

from __future__ import division

from sklearn import svm
from numpy import *
import numpy as np
import os


class SVMCharacterPredict:
    def myAcc(self, y_true, y_pred):
        """
        准确值计算
        :param y_true:
        :param y_pred:
        :return:
        """
        true_num = 0
        # 最大数的索引
        y_pred = np.argmax(y_pred, axis=1)

        # for i in range(y_true.__len__()):
        #     print y_true[i]
        for i in range(y_pred.__len__()):
            if y_true[i] == y_pred[i]:
                true_num += 1
        return true_num

    def mymean(self, list_predict_score, array_test):
        """
        my mean count
        :param list_predict_score:
        :param array_test:
        :return:
        """
        num_total = 0
        num_total = array_test.shape[0] * 5
        print "total numbers : " + str(num_total)
        return list_predict_score / (num_total)

    def train_eval(self, X_train, y_train, X_text, y_text):
        """
        输入矩阵 训练模型并计算准确率
        :param X_text:
        :param X_train:
        :param y_text:
        :param y_train:
        :return:
        """
        true_acc = 0
        for i in range(5):
            list_train_tags = []
            list_test_tags = []
            print "第" + str(i) + "个分类器训练"
            # first build train tag
            for line in y_train:
                list_train_tags.append(line[i])
            # first build text tag
            for line in y_text:
                list_test_tags.append(line[i])
            clf = svm.SVC(probability=True)
            clf = svm.SVC(kernel='linear', probability=True)
            # 逻辑回归训练模型
            clf.fit(X_train, list_train_tags)
            # 用模型预测
            y_pred_te = clf.predict_proba(X_text)
            # print np.argmax(y_pred_te, axis=1)
            # print "**" * 50
            # print list_test_tags
            # #获取准确的个数
            # print self.myAcc(list_test_tags, y_pred_te)
            true_acc += self.myAcc(list_test_tags, y_pred_te)
        print "true acc numbers: " + str(true_acc)
        return self.mymean(true_acc, X_text)

    def predict_by_textmind(self):
        """
        svm 文心特征
        :return:
        """
        X_train, Y_train, X_test, Y_test = input_textmind_data.load_textmind_data_label('../crawl_textmind_data')
        mymean = self.train_eval(X_train, Y_train, X_test, Y_test)
        print "textmind+支持向量机　准确率平均值为: " + str(mymean)

    def predict_by_d2v_dm(self):
        """
        d2v_dm 训练
        :return:
        """
        base_model_dir = ''
        train_vec_filename = os.path.join(base_model_dir, "doc2vec_train_vec_dm.npy")
        train_label_filename = os.path.join(base_model_dir, 'doc2vec_train_label_dm.npy')
        test_vec_filename = os.path.join(base_model_dir, 'doc2vec_test_vec_dm.npy')
        test_label_filename = os.path.join(base_model_dir, 'doc2vec_test_label_dm.npy')

        X_train, Y_train, X_test, Y_test = self.load_arr(test_label_filename, test_vec_filename, train_label_filename,
                                                         train_vec_filename)
        mymean = self.train_eval(X_train, Y_train, X_test, Y_test)
        print "d2v_dm+支持向量机　准确率平均值为: " + str(mymean)

    def predict_by_d2v_dbow(self):
        """
        d2v_dbow 训练
        :return:
        """
        base_model_dir = ''
        train_vec_filename = os.path.join(base_model_dir, "doc2vec_train_vec_dbow.npy")
        train_label_filename = os.path.join(base_model_dir, 'doc2vec_train_label_dm.npy')
        test_vec_filename = os.path.join(base_model_dir, 'doc2vec_test_vec_dbow.npy')
        test_label_filename = os.path.join(base_model_dir, 'doc2vec_test_label_dm.npy')

        X_train, Y_train, X_test, Y_test = self.load_arr(test_label_filename, test_vec_filename, train_label_filename,
                                                         train_vec_filename)
        mymean = self.train_eval(X_train, Y_train, X_test, Y_test)
        print "d2v_dbow+支持向量机　准确率平均值为: " + str(mymean)

    def predict_by_tfidf(self):
        """
        tfidf 训练
        :return:
        """
        base_model_dir = ''
        train_vec_filename = os.path.join(base_model_dir, "../character/tfidf_train_vec_tfidf.npy")
        train_label_filename = os.path.join(base_model_dir, 'doc2vec_train_label_dm.npy')
        test_vec_filename = os.path.join(base_model_dir, '../character/tfidf_test_vec_tfidf.npy')
        test_label_filename = os.path.join(base_model_dir, 'doc2vec_test_label_dm.npy')

        X_train, Y_train, X_test, Y_test = self.load_arr(test_label_filename, test_vec_filename, train_label_filename,
                                                         train_vec_filename)
        mymean = self.train_eval(X_train, Y_train, X_test, Y_test)
        print "tfidf+支持向量机　准确率平均值为: " + str(mymean)

    def load_arr(self, test_label_filename, test_vec_filename, train_label_filename, train_vec_filename):
        X_train = np.load(train_vec_filename)
        print('X_train', X_train.shape)
        Y_train = np.load(train_label_filename)
        print('Y_train', Y_train.shape)
        X_test = np.load(test_vec_filename)
        print('X_test', X_test.shape)
        Y_test = np.load(test_label_filename)
        print('Y_test', Y_test.shape)
        return X_train, Y_train, X_test, Y_test


from crawl_textmind_data import input_textmind_data

if __name__ == '__main__':
    user_predict = SVMCharacterPredict()
    user_predict.predict_by_d2v_dbow()
    user_predict.predict_by_d2v_dm()
    user_predict.predict_by_tfidf()
    # user_predict.predict_by_textmind()
