# coding=utf-8
# 完整的神经网络样例程序——回归预测
__author__ = 'gu'

import tensorflow as tf
from numpy.random import RandomState

# 定义训练数据的大小
batch_size = 8



# 在shape的一个维度使用None 可以方便使用不用的batch 大小，两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
# 回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 定义神经网络参数
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
# 定义神经网络前向传播算法的过程——简单的加权求和
y = tf.matmul(x, w1)

# 自定义损失函数和反向传播算法
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.select(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# 通过随机数生成模拟数据集
ram = RandomState(1)
dataset_size = 128
X = ram.rand(dataset_size, 2)
# print(X)
# 设置回归的正确值为两个输入的和加上一个随机量。之所以加上随机量是因为加了不可预测的噪音，否则不同损失函数的意义就不大了
# 因为不同损失函数都会在能完全预测正确的时候最低。
# 一般来说噪音为一个均值为0的小量，所以这里的噪音设置为-0.05~0.05
Y = [[x1 + x2 + ram.rand() / 10.0 - 0.05] for (x1, x2) in X]

# 创建会话运行程序
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
    print(sess.run(w1))
    """
    [[1.019347 ]
     [1.0428089]]
    """
