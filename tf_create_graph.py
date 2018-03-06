# coding=utf-8
# tf.Graph生成新的计算图
# 不同计算图上的张量和运算不会共享
__author__ = 'gu'

import tensorflow as tf

g1 = tf.Graph()

with g1.as_default():
    # 在g1图上定义变量V 初始化为0  shape 代表矩阵的维度
    v = tf.get_variable("v",initializer=tf.zeros_initializer(shape=[1]))

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v",initializer=tf.ones_initializer(shape=[1]))

# 在计算图g1中读取变量v
with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        print sess.run(tf.get_variable("v"))


# 在计算图g2中读取变量v
with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        print sess.run(tf.get_variable("v"))


# tf中的集合列表
print(tf.GraphKeys.VARIABLES)
print(tf.GraphKeys.TRAINABLE_VARIABLES)
print(tf.GraphKeys.SUMMARIES)
print(tf.GraphKeys.QUEUE_RUNNERS)
print(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)