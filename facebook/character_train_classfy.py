# coding=utf-8
"""
定义了神经网络的训练过程
"""

import os

import tensorflow as tf

import character_inference
import input_data

BATCH_SIZE = 10  # 一个训练batch中的训练数据个数，数字越小，训练过程越接近随机梯度下降
LEARNING_RATE_BASE = 0.01  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化在损失函数的系数
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
MODEL_SAVE_PATH = "character_model/classfy/"
MODEL_NAME = "character_model"

train_list_side, train_list_tag, text_list_side, text_list_tag = input_data.load_classfy_data()

TRAIN_NUM_EXAMPLES = DATASET_SIZE = len(train_list_side)  # 训练数据的总数


def train():
    x = tf.placeholder(tf.float32, [None, character_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = character_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, targets=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        DATASET_SIZE / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range(TRAINING_STEPS):

            # # 每次选取batch_size样本进行训练
            start = (i * BATCH_SIZE) % DATASET_SIZE
            end = min(start + BATCH_SIZE, DATASET_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: train_list_side[start:end],
                                                      y_: train_list_tag[start:end]})
            # 每次选取all_size样本进行训练
            # _, loss_value, step = sess.run([train_op, loss, global_step],
            #                                feed_dict={x: train_list_side,
            #                                           y_: train_list_tag})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
