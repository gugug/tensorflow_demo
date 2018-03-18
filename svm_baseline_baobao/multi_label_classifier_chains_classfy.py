# coding=utf-8
"""
分类器链（Classifier Chains）
"""
__author__ = 'gu'
# using classifier chains
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from skmultilearn.problem_transform import ClassifierChain

from character.input_data import load_data_label

X_train, Y_train, X_test, Y_test = load_data_label('/home/gu/PycharmProjects/tensorflow_demo/character/')


# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
classifier = ClassifierChain(GaussianNB())

# train
classifier.fit(X_train, Y_train)

# predict
predictions = classifier.predict(X_test)

print accuracy_score(Y_test, predictions)
