# coding=utf-8
"""
测试mnist数据集
"""
__author__ = 'gu'

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# 载入MNISR数据集，如果指定地址/path/to.MNIST_DATA下没有下载好的数据，那么tensorflow会自动更新
mnist = input_data.read_data_sets("/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/mnist_data")
# 打印训练数据大小 55000
print(mnist.train.num_examples)

# 打印验证数据大小 5000
print(mnist.validation.num_examples)

# 打印测试数据大小 10000
print(mnist.test.num_examples)

# 打印example 训练数据
print(mnist.train.images[0])
print(len(mnist.train.images[0]))

# 打印训练数据标签
print(mnist.train.labels)


# 从train中读取小部分作为一个训练batch

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
print(xs.shape)  # (100, 784)
print(ys.shape)  # (100,)
