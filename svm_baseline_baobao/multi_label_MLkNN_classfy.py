# coding=utf-8
"""
标签Powerset（Label Powerset）"""
__author__ = 'gu'
# using classifier chains
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import MLkNN

from character.input_data import load_data_label


# using Label Powerset

X_train, Y_train, X_test, Y_test = load_data_label('/home/gu/PycharmProjects/tensorflow_demo/character/')

classifier = MLkNN(k=20)

# train
classifier.fit(X_train, Y_train)

# predict
predictions = classifier.predict(X_test)

print accuracy_score(Y_test, predictions)
