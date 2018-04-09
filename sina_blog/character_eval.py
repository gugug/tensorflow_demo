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

MOVING_AVERAGE_DECAY = 0.99  # 活动平均衰减率
MODEL_SAVE_PATH = "character_model/"
MODEL_NAME = "character_model"
print(MODEL_SAVE_PATH)
# 加载的时间间隔。
EVAL_INTERVAL_SECS = 2

train_list_side, train_list_tag, text_list_side, text_list_tag = input_data.load_label()


def evaluate(character):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, character_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.int64, name='y-input')
        validate_feed = {x: text_list_side, y_: text_list_tag}

        y = character_inference.inference(x, None)
        # y = character_inference.inference_nlayer(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                    print("==========================================")
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    evaluate(character)
    # mymean([1, 2, 1, 1, 2])


if __name__ == '__main__':
    tf.app.run()
