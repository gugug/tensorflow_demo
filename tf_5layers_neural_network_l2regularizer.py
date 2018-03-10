# coding=utf-8
"""
5层神经网络带L2正则化的损失函数计算方法
"""
__author__ = 'gu'

import tensorflow as tf


def get_weight(shape, lamd):
    """
    获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为’losses‘的集合中
    :param shape: 维度——对应多少个输入和多少个输出
    :param lamd: 正则化项的权重
    :return: 神经网络边上的权重
    """
    # 生成一个变量 代表权重
    var = tf.Variable(tf.random_normal(shape=shape), dtype=tf.float32)
    # 将这个权重的L2正则化损失加入名称为’losses‘的集合中
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamd)(var))
    # 返回一层神经网络边上的权重
    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8

# 定义没一层网络中的节点数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)

# 这个变量维护前向传播时最深的层，开始时就是输入层
cur_layer = x
# 当前层的节点数
in_dimension = layer_dimension[0]

# 通过循环来生成5层全连接的神经网络结构
for i in range(1, n_layers):
    # layer_dimension[i]为下一层的节点个数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量，并把这个变量的L2正则化损失加入计算图上的集合
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))

    # 使用ReLU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前将下一层的节点个数更新为当前层节点个数
    in_dimension = layer_dimension[i]

# 计算刻画模型在训练数据上表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 将均方误差损失函数加入损失集合
tf.add_to_collection('losses', mse_loss)

# get_Collection 返回列表，这些元素就是损失函数的不同部分，将它们加起来就可以最终得到损失函数
loss = tf.add_n(tf.get_collection('losses'))



