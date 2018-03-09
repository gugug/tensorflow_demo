# coding=utf-8
# 损失函数正则化 解决过拟合的问题
__author__ = 'gu'

import tensorflow as tf

weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])

with tf.Session() as sess:
    """

    Returns:
    A function with signature `l2(weights, name=None)` that applies L2
    regularization.
    返回一个函数，这个函数可以计算一个给定参数的L2正则化项的值

    (|1|+|-2|+|-3|+|4|)*0.5=5 其中0.5为正则化项的权重
    """
    print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(weights)))

    """
    (1^2 + (-2)^2 +(-3)^2 + 4^2) /2 * 0.5 = 7.5
    """
    print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(weights)))



# ==优化带正则化的损失函数==========================
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义placeholder作为存放输入数据的地方,维度可以不指定 会根据数据集推导得出
x = tf.placeholder(dtype=tf.float32, shape=(3, 2), name="x-input")
y_ = tf.placeholder(dtype=tf.float32, shape=(None,), name="y-input")

# 前向传播算法
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# y_ 为正确答案 y为预测值
lad = 0.5  # 0.5为正则化项的权重
# weights 为需要计算正则化损失的函数
loss = tf.reduce_mean(tf.square(y_ - y) + tf.contrib.layers.l2_regularizer(lad)(weights))
