# coding=utf-8
"""测试tf的词向量训练api"""
__author__ = 'gu'

from tensorflow.contrib import learn
import numpy as np

max_document_length = 4
x_text = [
    'i love you',
    'me too'
]
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vocab_processor.fit(x_text)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print x
print next(vocab_processor.transform(['i me too'])).tolist()
