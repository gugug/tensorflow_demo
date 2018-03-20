# coding=utf-8
"""
定义看前向传播的过程以及神经网络中的参数
"""

import tensorflow as tf

# 神经网络相关参数
INPUT_NODE = 300  # 用户的特征维度
OUTPUT_NODE = 5  # 输出5个类别的性格
LAYER1_NODE = 500  # 隱藏层的节点数


def get_weight_variable(shape, regularizer):
    # 通过 tf.get_variable获取变量 和Variable 一样，在测试的时候会通过保存的模型来加载这些变量的取值。
    # 滑动平均变量重命名（影子变量），所以可以直接通过同样的变量名字取到变量本身
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        # 加入损失集合
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    # 声明第一层神经网络的变量并完成前向传播
    with tf.variable_scope('layer1'):
        # 生成隱藏层的参数
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        # 偏置设置为0.1
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
        # 使用ReLU的激活函数 去线性化
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 声明第二层神经网络的变量并完成前向传播
    with tf.variable_scope('layer2'):
        # 生成输出层的参数
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        layer2 = tf.matmul(layer1, weights) + biases

    # 返回最后的前向传播的结果
    return layer2
