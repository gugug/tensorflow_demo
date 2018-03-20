# coding=utf-8
"""
测试过程
"""
import character

__author__ = 'gu'

import time
import tensorflow as tf
import character_inference
import numpy as np
import input_data
from crawl_textmind_data import input_textmind_data

MOVING_AVERAGE_DECAY = 0.99  # 活动平均衰减率
MODEL_SAVE_PATH = "character_model/"
MODEL_NAME = "character_model"

# 加载的时间间隔。
EVAL_INTERVAL_SECS = 5


# 加载d2v 和 tfidf的数据
train_list_side, train_list_tag, test_list_side, test_list_tag = input_data.load_data_label('')
# 加载textmind的特征
# train_list_side, train_list_tag, test_list_side, test_list_tag = input_textmind_data.load_textmind_data_label('../crawl_textmind_data')

# 加载整合后的特征
# train_list_side, train_list_tag, test_list_side, test_list_tag = input_data.load_data_label_combine()

def evaluate(character):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, character_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.int64, name='y-input')
        validate_feed = {x: test_list_side, y_: test_list_tag}

        y = character_inference.inference(x, None)

        # accuracy = get_acc(y_, y)


        # correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state 会根据checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # accuracy_score = sess.run(accuracy, feed_dict=validate_feed)

                    # accuracy_score = get_acc(sess,true_y, pred_y)
                    # print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))

                    # print("the input data are \n%s" % test_list_side)
                    # print("the truly answer are \n%s" % test_list_tag)
                    eval_aws = sess.run(y, feed_dict=validate_feed)
                    # print("the evaluate answer are \n%s" % eval_aws)

                    accuracy_score = get_acc(sess, test_list_tag, eval_aws)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))

                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def get_acc(sess, true_y, pred_y):
    pred_y_ = np.where(pred_y > 0, 1, 0)
    correct_prediction = tf.equal(true_y, pred_y_)
    accuracy = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
    return accuracy


def main(argv=None):
    evaluate(character)  # 0.530223


if __name__ == '__main__':
    tf.app.run()
