# coding=utf-8
"""
通过tensorflow实现反响传播的第一步是使用tensorflow所表达一个batch样例，由于tensorflow是用计算图实现计算，
如果用常量表示，没生成一个常量增加一个节点，计算图会越来越大，所以tensorflow提供了placeholder机制，用于提供输入数据，
相当于定义了一个位置，这个位置的数据在运行时再指定。
所以不需要生成大量的常量来提供输入数据，而只需要讲数据通过placeholder传人计算图。这个位置的数据类型需要指定。
"""
__author__ = 'gu'

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义placeholder作为存放输入数据的地方,维度可以不指定 会根据数据集推导得出
x = tf.placeholder(dtype=tf.float32, shape=(3, 2), name="input")
a = tf.matmul(x, w1)

y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)

print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))



