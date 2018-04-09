# coding=utf-8
# 自定义损失函数
__author__ = 'gu'

import tensorflow as tf

v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0, 2.0, 1.0])

sess = tf.InteractiveSession()
print(tf.greater(v1, v2).eval())
"""对每一个元素比较大小
[False False  True  True]
"""

print(tf.select(tf.greater(v1, v2), v1, v2).eval())
"""
[4. 3. 3. 4.]
"""

# 执行自定义损失函数
loss = tf.reduce_sum(tf.select(tf.greater(v1, v2), v1, v2).eval())
print(loss.eval())
"""
14.0
"""
sess.close()
