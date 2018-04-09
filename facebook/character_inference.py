# coding=utf-8
"""
定义看前向传播的过程以及神经网络中的参数
"""

import math

import tensorflow as tf

INPUT_NODE = 102  # 用户的特征维度
OUTPUT_NODE = 5  # 输出5个类别的性格
# LAYER1_NODE = 8  # 隱藏层的节点数 根据经验公式lgn
expr = 0.43 * INPUT_NODE * 5 + 0.12 * 5 * 5 + 2.54 * INPUT_NODE + 0.77 * 5 + 0.35
LAYER1_NODE = int(math.sqrt(expr) + 0.51)


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    """
    一层隱藏层神经网络前向传播算法
    :param input_tensor:
    :param regularizer:
    :return:
    """
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


def get_weight(shape, regularizer):
    """
    获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为’losses‘的集合中
    :param shape: 维度——对应多少个输入和多少个输出
    :param lamd: 正则化项的权重
    :return: 神经网络边上的权重
    """
    var = tf.Variable(tf.random_normal(shape=shape), dtype=tf.float32)
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(var))
    return var


def inference_nlayer(input_tensor, regularizer):
    """
    n层神经网络前向传播算法
    :param input_tensor:
    :param regularizer:
    :return:
    """
    layer_dimension = [INPUT_NODE, 100, 100, 100, OUTPUT_NODE]
    n_layers = len(layer_dimension)
    cur_layer = input_tensor
    in_dimension = layer_dimension[0]

    for i in range(1, n_layers):
        out_dimension = layer_dimension[i]
        weight = get_weight([in_dimension, out_dimension], regularizer)
        bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))

        cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
        in_dimension = layer_dimension[i]
    return cur_layer
