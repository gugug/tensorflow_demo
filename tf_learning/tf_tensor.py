# coding=utf-8
# 张量的概念和使用
__author__ = 'gu'

# 张量中并没有真正保存数字，只是保存的是如何得到这些数字的计算过程
import tensorflow as tf

a = tf.constant([1.0,2.0],name='a')
b = tf.constant([2.0,3.0],name='b')

result = tf.add(a,b,name="add")
print(result)  # Tensor("add:0", shape=(2,), dtype=float32)
print(tf.Session().run(result))  # [3. 5.]

"""
 Tensor("add:0", shape=(2,), dtype=float32)
 输出的结果不是计算结果，而是张量的结构，有三个属性：名字(name) 维度(shape) 数据类型(dtype)
 属性1: name 不仅表示张量的唯一标识符，更是记录了如何计算出来的。node:src_output,node 表示节点的名称，src_output表示当前节点的第几个输出(从0开始)
 属性2: shape 表示张量的维度，shape=(2,)说明是一维数组长度为2
 属性3: dtype 每个张量都有对应的类型
"""