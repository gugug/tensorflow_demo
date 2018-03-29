# -*- coding: UTF-8 -*-
"""tfidf-lsi-svm stack for character"""

from __future__ import division

import codecs
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.externals import joblib
from numpy import *
from gensim import models, corpora
import numpy as np


class user_predict:
    def __init__(self, train_document, text_document):
        self.train_document = train_document
        self.text_document = text_document

    # -----------------------准确值计算-----------------------
    def myAcc(self, y_true, y_pred):
        true_num = 0
        # 最大数的索引
        y_pred = np.argmax(y_pred, axis=1)

        # for i in range(y_true.__len__()):
        #     print y_true[i]
        for i in range(y_pred.__len__()):
            if y_true[i] == y_pred[i]:
                true_num += 1
        return true_num

    # -----------------------load data-----------------------
    def load_data(self, doc):

        list_name = []  # id
        list_total = []  # 文本
        list_label = []  # 标签
        # 对应标签导入词典
        f = codecs.open(doc)
        temp = f.readlines()
        f.close()
        for i in range(len(temp)):
            temp[i] = temp[i].split(" ")
            user_name = temp[i][0]
            tags = temp[i][1:6]
            query = temp[i][6:]
            query = " ".join(query).strip().replace("\n", "")

            list_total.append(query)
            list_label.append(tags)

        # 字符串标签转化为int类型
        list_tag = []
        for line in list_label:
            list_t = []
            for j in line:
                j = int(j)
                list_t.append(j)
            list_tag.append(list_t)

        print "data have read "
        return list_total, list_tag

    # -------------------------prepare lsi svd -----------------------
    def prepare(self, doc):

        # 给训练集用的，返回文本和对应的标签
        list_total, list_tag = self.load_data(doc)

        stop_word = []
        texts = [[word for word in document.lower().split()]
                 for document in list_total]

        dictionary = corpora.Dictionary(texts)  # 生成词典
        tfv = TfidfVectorizer(min_df=1, max_df=0.95, sublinear_tf=True, stop_words=stop_word)
        X_sp = tfv.fit_transform(list_total)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf_model = models.TfidfModel(corpus)
        joblib.dump(tfidf_model, "tfidf_model_notlsi.model")
        joblib.dump(dictionary, "tfidf_dictionary_notlsi.dict")
        return tfidf_model, dictionary, X_sp

    def train(self, doc):
        list_total, list_tag = self.load_data(doc)
        tfv = TfidfVectorizer(min_df=1, max_df=0.95, sublinear_tf=True, stop_words=[])
        X_sp = tfv.fit_transform(list_total)
        print X_sp.shape
        return list_total, list_tag, X_sp

    # ------------------------my mean count------------------
    def mymean(self, list_predict_score, array_test):
        num_total = 0
        num_total = array_test.shape[0] * 5
        print "total numbers : " + str(num_total)
        return list_predict_score / (num_total)

    # ------------------------------begin to predict------------
    def predict(self):
        train_list_total, train_list_tag, train_list_side = self.train(self.train_document)
        print "train model done -------------------"
        text_list_total, text_list_tag, text_list_side = self.train(self.text_document)
        print "text model done  -------------------"
        X_train = train_list_side
        y_train = train_list_tag
        y_train = np.array(y_train)
        print "train shape :---------------------"
        print X_train.shape
        X_text = text_list_side
        y_text = text_list_tag
        y_text = np.array(y_text)
        print "text shape :---------------------"
        print X_text.shape
        true_acc = 0
        for i in range(5):
            list_train_tags = []
            list_test_tags = []
            print "第" + str(i) + "个分类器训练"
            for line in y_train:
                list_train_tags.append(line[i])
            for line in y_text:
                list_test_tags.append(line[i])
            clf = svm.SVC(probability=True)
            clf = svm.SVC(kernel='linear', probability=True)
            clf.fit(X_train, list_train_tags)
            y_pred_te = clf.predict_proba(X_text)
            print np.argmax(y_pred_te, axis=1)
            print "**" * 50
            print list_test_tags

            # #获取准确的个数
            print self.myAcc(list_test_tags, y_pred_te)
            true_acc += self.myAcc(list_test_tags, y_pred_te)
        print "true acc numbers: " + str(true_acc)
        print "不使用LSI降维 + 支持向量机　准确率平均值为: "
        print self.mymean(true_acc, X_text)


if __name__ == '__main__':
    base_dir = '/home/gu/PycharmProjects/tensorflow_demo/essay_data'
    user_predict = user_predict(os.path.join(base_dir, "vocab1_train.txt"), os.path.join(base_dir, "vocab1_test.txt"))
    user_predict.predict()
