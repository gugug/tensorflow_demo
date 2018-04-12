# coding=utf-8
"""
测试过程
"""

__author__ = 'gu'

import time
import tensorflow as tf
import character_inference
import numpy as np
import input_data

MOVING_AVERAGE_DECAY = 0.99  # 活动平均衰减率
MODEL_SAVE_PATH = "character_model/"
MODEL_NAME = "character_model"
print(MODEL_SAVE_PATH)
# 加载的时间间隔。
EVAL_INTERVAL_SECS = 2

train_list_side, text_list_side = input_data.load_x()
train_list_tag, text_list_tag = input_data.load_liner_y()


def evaluate():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, character_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, name='y-input')

        y = character_inference.inference(x, None)
        mse = tf.reduce_mean(tf.square(y_ - y))

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                validate_feed = {x: text_list_side, y_: text_list_tag[:, 1]}

                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # eval_aws = sess.run(y, feed_dict=validate_feed)
                    loss = sess.run(mse, feed_dict=validate_feed)
                    print("After %s training step(s) loss %s" % (global_step, loss))
                    print("==========================================")
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.run()
