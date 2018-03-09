# coding=utf-8
# 分类问题和回归问题中损失函数——均方误差 ＭＳＥ
__author__ = 'gu'

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义placeholder作为存放输入数据的地方,维度可以不指定 会根据数据集推导得出
x = tf.placeholder(dtype=tf.float32, shape=(3, 2), name="x-input")
y_ = tf.placeholder(dtype=tf.float32, shape=(None,), name="y-input")

# 前向传播算法
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 均方误差ＭＳＥ y_ 为正确答案 y为预测值
mse = tf.reduce_mean(tf.square(y_ - y))
