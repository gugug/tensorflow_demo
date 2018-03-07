# coding=utf-8
# 运行模式 session
__author__ = 'gu'

import tensorflow as tf

a = tf.constant([1.0,2.0],name='a')
b = tf.constant([2.0,3.0],name='b')

result = tf.add(a,b,name="add")

# 显式创建会话和显式关闭
sess = tf.Session()
print(result.eval(session=sess))
print(sess.run(result))
sess.close()

# 创建上下文管理器使用会话
# 不需要显式创建会话和显式关闭，上下文退出的时候自动完成
with tf.Session() as sess:
    print sess.run(result)

# 类似于计算图 也有默认的会话，但是不会自动生成默认的会话。需要手动指定
sess1 = tf.Session()
with sess1.as_default():
    print(result.eval())

# 交互环境下直接构建默认会话
inter_sess = tf.InteractiveSession()
print(result.eval())
inter_sess.close()

# 配置会话
config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
"""
allow_soft_placement 默认为false, 作用是GPU上的运算可以放到CPU上进行（
    如果运算无法在GPU上执行
    没有GPU资源
    运算输入包含对CPU的引用）
可移植性高
log_device_placement 日志记录中将会记录每个节点安排在哪个设备上
"""
sess_conf1 = tf.InteractiveSession(config=config)
sess_conf2 = tf.Session(config=config)
