# coding=utf-8

# forward propagation 前向传播
__author__ = 'gu'

import tensorflow as tf

# 声明两个变量
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# 输入特征变量  一行两列
input_x = tf.constant([[0.7,0.9]])

sess = tf.Session()
# 初始化变量
sess.run(w1.initializer)
sess.run(w2.initializer)

# 前向传播算法计算神经网络的输出
a = tf.matmul(input_x,w1)
print(sess.run(a))

y = tf.matmul(a,w2)

print(sess.run(input_x))

# 运行输出
print sess.run(y)

sess.close()