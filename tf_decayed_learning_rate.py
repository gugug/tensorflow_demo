# coding=utf-8
# 指数衰减法设置学习率
"""
指数衰减法设置学习率

在训练神经网络时，需要设置学习率控制参数更新的速度。学习率决定了参数每次更新的幅度，
如果幅度过大，可能导致参数在极优值的两侧来回移动。
学习率过小，虽然能够保证收敛性没但是这会大大降低优化速度。
tensorflow提供了一个更加灵活的学习率设置方法——指数衰减法。
通过这个方法，可以先使用较大的学习率来快速得到一个比较优的解，然后随着迭代的继续逐步减少学习率。
"""
__author__ = 'gu'

import tensorflow as tf

global_step = tf.Variable(0)

# 定义神经网络大参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 在shape的一个维度使用None 可以方便使用不用的batch 大小
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
# 定义神经网络前向传播算法的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 通过tf.train.exponential_decay 函数生成学习率
"""
因为staircase为True,所以每训练100轮后学习率乘以0.96

实现了一下功能：
decayed_learning_rate = learning_rate * decay_rate^(global_step / decay_steps)


Args:
    learning_rate:  The initial learning rate. 事先设定的学习率
    global_step: 衰减速度
    decay_steps: 衰减系数
    decay_rate: 衰减系数
    staircase: 默认为false， 当为True的时候，学习率成为一个阶梯函数
    name: String.  Optional name of the operation.  Defaults to 'ExponentialDecay'

"""
learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step,
                                           decay_steps=100, decay_rate=0.96, staircase=True)

# 定义损失函数和反向传播算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

# 使用衰减指数的学习率，在minimize函数中传入global_step将自动更新global_step参数，从而使得学习率也得到相应更新
tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy, global_step=global_step)
