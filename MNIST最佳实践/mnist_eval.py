# coding=utf-8
__author__ = 'gu'

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train


# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.int64, name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state 会根据checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/mnist_data")
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
