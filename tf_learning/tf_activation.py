# coding=utf-8
# 使用激活函数进行线性变换
__author__ = 'gu'

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.constant([[0.7, 0.9]])

biases1 = 1
biases2 = 2
# 使用激活函数线性变换
"""
tf.nn.relu()
tf.tanh()
tf.sigmoid()
"""
a = tf.nn.relu(tf.matmul(x, w1) + biases1)
y = tf.nn.relu(tf.matmul(a, w2) + biases2)

sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)

sess.run(y)

sess.close()
