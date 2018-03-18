# coding=utf-8
"""
二元关联（Binary Relevance）
"""
__author__ = 'gu'
# using binary relevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from skmultilearn.problem_transform import BinaryRelevance

from character.input_data import load_data_label


# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())

X_train, Y_train, X_test, Y_test = load_data_label('/home/gu/PycharmProjects/tensorflow_demo/character/')

# train
classifier.fit(X_train, Y_train)

# predict
predictions = classifier.predict(X_test)

print accuracy_score(Y_test, predictions)
