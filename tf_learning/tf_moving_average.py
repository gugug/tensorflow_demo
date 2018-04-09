# coding=utf-8
"""
滑动平均模型
tf.train.ExponentialMovingAverage 需要指定一个衰减率decay,这个衰减率用于控制模型更新的速度
对每一个变量维护一个影子变量（shadow_variable）,这个影子变量的初始值就是相应变量的初始值，
而每次运行变量更新时，影子变量的更新公式：
shadow_variable = decay * shadow_variable + (1 - decay)*variable
decay 越大模型越趋于稳定

tf.train.ExponentialMovingAverage 还提供了num_updates 参数来动态设置decay的大小
衰减率为 min{decay, (1+num_updates)/(10+num_updates)}
"""
__author__ = 'gu'

import tensorflow as tf

# 定义一个变量用于计算滑动平均，这个变量初始值为0
v1 = tf.Variable(0, dtype=tf.float32)

# 这个step变量类似于模拟神经网络中迭代的次数，可以用于动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类，初始化时给顶衰减率和控制衰减率的变量
ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=step)

# 定义一个更新变量滑动平均的操作。这里需要给定一个列表，每次执行这个操作的时候，都会更新这个个列表中的变量
# 例如下面的列表中，每次执行这个maintain_average_op的操作都会更新列表中的变量v1
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    # 通过ema.average(v1)获取滑动平均之后变量的取值。在初始化之后变量v1的值和v1的滑动平均一样都为0
    print(sess.run([v1, ema.average(v1)]))

    # 更性v1的值为5
    sess.run(tf.assign(v1, 5))

    # 更新v1的滑动平均值。衰减率为min{0.99,(1+step)/(10+step)=0.1} = 0.1
    # 所以v1的滑动平均值会更新为0.1*0+（1-0.1）*5=4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 再次更新胡敖东平均值
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
