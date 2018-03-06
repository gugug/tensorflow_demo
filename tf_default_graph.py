# coding=utf-8

# 获取默认计算图
__author__ = 'gu'

import tensorflow as tf

a = tf.constant([1.0,2.0],name='a')
b = tf.constant([2.0,3.0],name='b')

result = a + b

sess = tf.Session()
print(sess.run(result))

# 通过a.graph可以查看张良所属的计算图a

print(a.graph is tf.get_default_graph())
