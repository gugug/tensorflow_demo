# -*- coding: UTF-8 -*-
'''tfidf-lsi-svm stack for gender'''

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

        list_name = []
        list_total = []
        list_gender = []
        # 对应标签导入词典
        f = codecs.open(doc)
        temp = f.readlines()
        print len(temp)

        for i in range(len(temp)):
            temp[i] = temp[i].split(" ")
            user_name = temp[i][0]
            tags = temp[i][1:6]

            query = temp[i][6:]
            query = " ".join(query).strip().replace("\n", "")
            # qu biao dian
            # query = re.sub("[+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"),query)

            list_total.append(query)
            list_gender.append(tags)

        print list_total.__len__()
        print list_gender.__len__()
        list_tag = []
        for line in list_gender:
            list_t = []
            for j in line:
                j = int(j)
                list_t.append(j)
            list_tag.append(list_t)

        print "data have read "
        return list_total, list_tag

    # -------------------------prepare lsi svd -----------------------
    def prepare_lsi(self, doc):

        # 给训练集用的
        list_total, list_tag = self.load_data(doc)

        stop_word = []

        texts = [[word for word in document.lower().split() if word not in stop_word]
                 for document in list_total]

        # train dictionary done
        dictionary = corpora.Dictionary(texts)  # 生成词典
        # print dictionary.token2id
        # 用TFIDF的方法计算词频,sublinear_tf 表示学习率
        tfv = TfidfVectorizer(min_df=1, max_df=0.95, sublinear_tf=True, stop_words=stop_word)
        # 对文本中所有的用户对应的所有的评论里面的单词进行ＴＦＩＤＦ的计算，找出每个词对应的tfidf值
        X_sp = tfv.fit_transform(list_total)
        corpus = [dictionary.doc2bow(text) for text in texts]
        # train model done
        tfidf_model = models.TfidfModel(corpus)
        joblib.dump(tfidf_model, "tfidf_model.model")

        corpus_tfidf = tfidf_model[corpus]

        lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)

        joblib.dump(dictionary, "tfidf_dictionary.dict")
        print "训练集lsi -----"
        joblib.dump(lsi_model, "tfidf_lsi.model")

        return tfidf_model, dictionary

    def train_lsi(self, doc, str_doc):

        if not (os.path.exists("tfidf_model.model")):

            print "prepare model"
            # load train model
            tfidf_model, dictionary = self.prepare_lsi(doc)

            # load data
            list_total, list_tag = self.load_data(doc)
            stop_word = []
            texts = [[word for word in document.lower().split() if word not in stop_word]
                     for document in list_total]

            corpus = [dictionary.doc2bow(text) for text in texts]

        else:
            print "use model"
            # load train valid text
            tfidf_model = joblib.load("tfidf_model.model")
            dictionary = joblib.load("tfidf_dictionary.dict")

            # load data
            list_total, list_tag = self.load_data(doc)
            stop_word = []
            texts = [[word for word in document.lower().split() if word not in stop_word]
                     for document in list_total]

            corpus = [dictionary.doc2bow(text) for text in texts]

        lsi_model = joblib.load("tfidf_lsi.model")
        corpus_tfidf = tfidf_model[corpus]

        list_side = []

        corpus_lsi = lsi_model[corpus_tfidf]
        nodes = list(corpus_lsi)

        for i in range(len(nodes)):
            list_d = []
            for j in range(200):
                # print nodes[i][j]
                list_d.append(nodes[i][j][1])
            list_side.append(list_d)

        list_vec = mat(list_side)
        self.write_d2v(list_vec, str_doc)
        print "lsi 矩阵构建完成----------------"

        return list_total, list_tag, list_side

    # -----------------------write vec--------------------
    def write_d2v(self, X_sp, doc_name):

        file_name = "tfidf_" + doc_name + ".npy"

        np.save(file_name, X_sp)
        print "*****************write done over *****************"

    # ------------------------my mean count------------------
    def mymean(self, list_predict_score, array_test):
        num_total = 0
        num_total = array_test.shape[0] * 5
        print "total numbers : " + str(num_total)
        return list_predict_score / (num_total)

    # ------------------------------begin to predict------------
    def predict(self):
        str1 = "train_vec_tfidf"
        str2 = "test_vec_tfidf"

        train_list_total, train_list_tag, train_list_side = self.train_lsi(self.train_document, str1)
        print "train model done -------------------"

        text_list_total, text_list_tag, text_list_side = self.train_lsi(self.text_document, str2)
        print "text model done  -------------------"

        TR = train_list_total.__len__()
        TE = text_list_total.__len__()
        n = 5

        train_list_side = mat(train_list_side)
        text_list_side = mat(text_list_side)

        X_train = train_list_side[:TR]
        y_train = train_list_tag[:TR]
        y_train = np.array(y_train)

        print "train shape :---------------------"
        print X_train.shape

        X_text = text_list_side[:TE]
        y_text = text_list_tag[:TE]
        y_text = np.array(y_text)

        print "text shape :---------------------"
        print X_text.shape

        # kfold折叠交叉验证
        list_myAcc = []
        self.train_eval(X_train, y_train, X_text, y_text)

    def train_eval(self, X_train, y_train, X_text, y_text):
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

            print np.argmax(y_pred_te, axis=1)
            print "**" * 50
            print list_test_tags

            # #获取准确的个数
            print self.myAcc(list_test_tags, y_pred_te)
            true_acc += self.myAcc(list_test_tags, y_pred_te)
        print "true acc numbers: " + str(true_acc)
        print "LSI + 支持向量机　准确率平均值为: "
        print  self.mymean(true_acc, X_text)


if __name__ == '__main__':
    base_dir = '/home/gu/PycharmProjects/tensorflow_demo/essay_data'
    user_predict = user_predict(os.path.join(base_dir, "vocab1_train.txt"), os.path.join(base_dir, "vocab1_test.txt"))
    user_predict.predict()
