# coding=utf-8
# 分类问题中损失函数——交叉熵 刻画概率分布之间的距离
__author__ = 'gu'

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义placeholder作为存放输入数据的地方,维度可以不指定 会根据数据集推导得出
x = tf.placeholder(dtype=tf.float32, shape=(3, 2), name="x-input")
y_ = tf.placeholder(dtype=tf.float32, shape=(None,), name="y-input")

# 前向传播算法
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 交叉熵运算
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, clip_value_min=1e-10, clip_value_max=1.0)))

# 与softmax回归一起使用
softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
"""
y_ 表示的是正确结果，y表示的预测结果，

tf.clip_by_value函数可以将一个张量中的数值限制在一个范围之内，可以避免一些运算错误（log0无效等）。
  Given a tensor `t`, this operation returns a tensor of the same type and
  shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
  Any values less than `clip_value_min` are set to `clip_value_min`. Any values
  greater than `clip_value_max` are set to `clip_value_max`.

tf.log 函数表示对张量中所有元素一次求对数的功能

* 表示乘法，不是矩阵乘法而是元素之间的直接相乘，矩阵乘法用tf.matmul

tf.reduce_mean 函数对张量进行每行中m个结果相加得到所有样例的交叉熵，然后在对这n行取平均得到一个batch的平均交叉熵
"""

# ============交叉熵运算中函数的一些测试===============================
sess = tf.Session()
with sess.as_default():
    test_clip_value = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(tf.clip_by_value(test_clip_value, 2.5, 4.5).eval())
    """小于2.5的换成2.5 大于4.5的换成4.5
        [[2.5 2.5 3. ]
         [4.  4.5 4.5]] """
    test_log_value = tf.constant([1.0, 2.0, 3.0])
    print(tf.log(test_log_value).eval())
    """对每一个元素求对数
        [0.        0.6931472 1.0986123]"""

    test_element_multi1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    test_element_multi2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    print (test_element_multi1 * test_element_multi2).eval()
    """* 表示直接元素之间相乘
    [[ 5. 12.]
     [21. 32.]]
    """
    print(tf.matmul(test_element_multi1, test_element_multi2).eval())
    """矩阵相乘
    [[19. 22.]
     [43. 50.]]
    """

    test_reduce_mean_value = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(tf.reduce_mean(test_reduce_mean_value).eval())
    """整个矩阵做平均
    3.5
    """
