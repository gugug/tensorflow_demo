# coding=utf-8
# 完整的神经网络样例程序——二分类问题
__author__ = 'gu'

import tensorflow as tf
from numpy.random import RandomState

# 定义训练数据的大小
batch_size = 8

# 定义神经网络大参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在shape的一个维度使用None 可以方便使用不用的batch 大小
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 定义神经网络前向传播算法的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

# 通过随机数生成模拟数据集
ram = RandomState(1)
dataset_size = 128
X = ram.rand(dataset_size, 2)
# print(X)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建会话运行程序
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    """
    在训练之前神经网络的参数值
    w1 = [[-0.8113182   1.4845988   0.06532937]
          [-2.442704    0.0992484   0.5912243 ]]

    w2 = [[-0.8113182 ]
          [ 1.4845988 ]
          [ 0.06532937]]
    """

    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

            """
            After 0 training step(s), cross entropy on all data is 0.0674925
            After 1000 training step(s), cross entropy on all data is 0.0163385
            After 2000 training step(s), cross entropy on all data is 0.00907547
            After 3000 training step(s), cross entropy on all data is 0.00714436
            After 4000 training step(s), cross entropy on all data is 0.00578471
            可以看出随着训练的进行，交叉熵逐渐变小，索命预测结果和真实值差距越小
            """
    print(sess.run(w1))
    print(sess.run(w2))
    """
    训练之后神经网络参数的值
    w1 = [[-1.9618274  2.582354   1.6820378]
          [-3.4681716  1.0698233  2.11789  ]]
    w2 = [[-1.8247149]
          [ 2.6854665]
          [ 1.418195 ]]
    已经发生变化，这个就是训练的结果，使得神经网络更好的拟合提供的训练数据
    """

"""
==========总结训练神经网络的过程3个步骤==========================
1 定义神经网络的结构和前向传播的输出结果
2 定义损失函数以及选择反向传播优化的算法
3 生成会话并且在训练数据上反复运行反向传播优化算法
"""
