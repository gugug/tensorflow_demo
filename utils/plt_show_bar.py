# coding=utf-8
"""show bar by plt"""
import matplotlib.pyplot as plt
import numpy as np


def drawPillar():
    n_groups = 10
    means_SVM = (0.545548794, 0.55100293, 0.536398467, 0.584584179, 0.558846067,
                 0.559927879, 0.585936444, 0.549244985, 0.569934641, 0.580166779)
    means_DNN = (0.553753, 0.5699797, 0.538742, 0.5979716, 0.5488844,
                 0.5533469, 0.53995943, 0.56754565, 0.568357, 0.5878296)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    rects1 = plt.bar(index, means_SVM, bar_width, alpha=opacity, color='b', label='SVM')
    rects2 = plt.bar(index + bar_width, means_DNN, bar_width, alpha=opacity, color='r', label='DNN')

    plt.xlabel('feature')
    plt.ylabel('acc')
    plt.title('acc by feature')

    x_label = ('emotion', 'textmind', 'd2v_dbow', 'tfidf',
               'emotion + textmind', 'emotion + dbow', 'emotion + tfidf', 'textmind + dbow',
               'textmind + tfidf', 'dbow + tfidf')
    plt.xticks(index + bar_width, x_label, rotation=90)

    plt.ylim(0, 0.9)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    drawPillar()
