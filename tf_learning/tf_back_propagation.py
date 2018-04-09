# coding=utf-8
"""
在得到一个batch的前向传播结果之后，需要定义一个损失函数来刻画当前的预测值和真实答案之间的差距。
然后通过反向传播算法来调整神经网络参数的取值使得差距越来越小。
"""
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

sess = tf.Session()
# 初始化变量
init_op = tf.initialize_all_variables()
sess.run(init_op)

# 定义损失函数来刻画预测值和真实值的差距
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

# 定义学习率
learning_rate = 0.001
# 定义反向传播算法来优化神经网络中的参数w1 w2
"""
常用的优化算法
tf.train.AdamOptimizer
tf.train.GradientDescentOptimizer
tf.train.MomentumOptimizer
"""
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

print(sess.run(train_step, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]], y_: []}))
print(sess.run(cross_entropy, feed_dict={x: [], y_: []}))
sess.close()
