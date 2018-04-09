# coding=utf-8
"""
tensorflow 完整程序解决mnist手写数字实体识别问题
"""
__author__ = 'gu'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# MNIST 数据集相关的常数
INPUT_NODE = 784  # 输入层的节点数，每一张图片是一个长度为784的一维矩阵
OUTPUT_NODE = 10  # 输出层的节点数，对应0~9

# 配置神经网络的参数
LAYER1_NODE = 500  # 隱藏层节点数，这里使用只有一层隱藏层的网络结构500个节点
BATCH_SIZE = 100  # 一个训练batch中的训练数据个数，数字越小，训练过程越接近随机梯度下降
LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化在损失函数的系数
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 活动平均衰减率


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    """
    辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。
    定义一个ReLU激活函数的三层全连接神经网络。通过加入隱藏层实现多层网络结构
    :param input_tensor:
    :param avg_calss: 用于计算参数平均值的类
    :param weights1:
    :param biases1:
    :param weights2:
    :param biases2:
    :return: 神经网络的前向传播结果
    """
    # 如果没有提供滑动平均类没直接使参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果，使用Ｒelu激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # 计算输出层的前向传播结果
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 首先使用avg_class.average计算变量的滑动平均值
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1))
            + avg_class.average(biases1)
        )
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    """
    训练模型的过程
    :param mnist: 处理ＭＮＩＳＴ数据集的类
    :return:
    """
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.int64, name='y-input')

    # 生成隱藏层的参数
    weights1 = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(value=0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(value=0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下神经网络前向传播的结果，滑动平均类为None
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的便利那个。这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量
    global_step = tf.Variable(0, trainable=False)

    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, num_updates=global_step)


    # 在所有代表神经网络参数的变量上使用滑动平均。tf.trainable_variable 返回的就是图上的集合tf.GraphKeys.TRAINABLE_VARIABLES的元素
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用滑动平均之后的前向传播结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵作为刻画预测值和真实值之间茶军的损失函数
    """
    This op expects unscaled logits, since it performs a softmax
      on `logits` internally for efficiency.  Do not call this op with the
      output of `softmax`
    第一个参数是神经网络不包括softmax层的前向传播结果，第二个参数是训练数据的正确答案的数字。
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
    # 计算在昂钱batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失，一般只计算神经网络边上权重的正则化损失，而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
        global_step,  # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY  # 学习率衰减速度
    )

    # 优化损失函数,在minizer中传入global_step将自动更新global_step,从而更新学习率
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时既需要通过反向传播来更新神经网络的参数，又要更新每一个参数的滑动平均值。
    # tf.control_dependencies和tf.group两种机制实现一次完成多个操作
    # train_op = tf.group(train_step, variable_averages_op)  # 等价下面两行
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 检验使用滑动平均模型的神经网络前向传播是否正确
    # average_y 是一个batch_size*10的二维数组，每一行表示一个样例的前向传播结果
    correct_prediction = tf.equal(tf.argmax(average_y, 1), y_)
    # 先将布尔型的数值转换为实数型，然后计算平均值，这个平均值就是一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        # 准备验证数据。
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代训练神经网络
        for i in range(TRAINING_STEPS):

            # 产生这一抡使用一个batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

        # 在训练结束之后，在测试数据上检测神经网络的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g " % (TRAINING_STEPS, test_acc))


# 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/mnist_data")
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
