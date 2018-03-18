# coding=utf-8
"""
标签Powerset（Label Powerset）"""
__author__ = 'gu'
# using classifier chains
from sklearn.metrics import accuracy_score

from character.input_data import load_data_label


# using Label Powerset
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

X_train, Y_train, X_test, Y_test = load_data_label('/home/gu/PycharmProjects/tensorflow_demo/character/')

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())

# train
classifier.fit(X_train, Y_train)

# predict
predictions = classifier.predict(X_test)

print accuracy_score(Y_test, predictions)
