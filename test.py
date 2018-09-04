# coding=utf-8
__author__ = 'gu'

import tensorflow as tf
import numpy as np

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')


def test(argv=None):
    logits = [[0.5, 0.7, 0.3, 0], [0.8, 0, 0.2, 0.9]]
    labels = tf.ones_like(logits)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
    sess = tf.Session()
    print(sess.run(labels))
    print sess.run(loss)


def test1(argv=None):
    test_label = [[0, 1, 1, 0, 1], [1, 0, 0, 1, 0]]
    for ty in test_label:
        # 最大数的索引
        print np.sum(ty)
    print np.sum(test_label, axis=1)


def top_k(true_y, pred_y):
    with tf.Session() as sess:
        true_num = 0
        elems_len = np.sum(true_y, axis=1)
        for idx in range(len(pred_y)):
            print elems_len[idx]
            leng = elems_len[idx]
            pred_idx = tf.nn.top_k(pred_y[idx], leng)[1]
            true_idx = tf.nn.top_k(true_y[idx], leng)[1]
            print('pred_idx', sess.run(pred_idx))
            print('true_idx', sess.run(true_idx))
            correct_prediction = tf.equal(pred_idx, true_idx)
            corr_num = sess.run(tf.cast(correct_prediction, tf.float32))
            print(corr_num)
            true_num += np.sum(corr_num)
            print('true_num', true_num)
        total_num = len(true_y) * 5
        print('acc', true_num * 1.0 / total_num)
        return true_num * 1.0 / total_num


if __name__ == '__main__':
    # tf.app.run()
    import math

    input = 600
    expr = 0.43 * input * 5 + 0.12 * 5 * 5 + 2.54 * input + 0.77 * 5 + 0.35
    print int(math.sqrt(expr) + 0.51)

    pred_y = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    pred_y = map(list, zip(*pred_y))
    print(pred_y)