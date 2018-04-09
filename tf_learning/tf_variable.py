# coding=utf-8
# 随机数生成函数方法和常数生成函数

__author__ = 'gu'

import tensorflow as tf
"""
生成一个2*3的矩阵 矩阵中的元素均值为1.0(默认为0) 标准差为2.0的随机数
随即生成函数
tf.random_normal()
tf.truncated_normal()
tf.random_uniform()
"""

var_random = tf.Variable(tf.random_normal(shape=[2,3],mean=1.0, stddev=2.0))
with tf.Session() as sess:
    print sess.run(var_random.initial_value)


# 常数生成函数
arr_zero = tf.zeros([2,3])
arr_one = tf.ones(shape=[2,3])
arr_nine = tf.fill([2,3],9)
arr_const = tf.constant([1,2,3])
cons = tf.constant(0.1,shape=[3])
with tf.Session() as sess:
    print sess.run(arr_zero)
    print sess.run(arr_one)
    print sess.run(arr_nine)
    print sess.run(arr_const)
    print(sess.run(cons))

# 通过其他变量的初始值来初始化新的变量
tf.Variable(var_random.initial_value)
