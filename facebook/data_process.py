# coding=utf-8

import os
from collections import defaultdict
import re
import csv

import numpy as np
import pandas as pd

__author__ = 'gu'


def readcsv(datafile):
    """
    load fb csv
    """
    print('loading csv...')
    print('loading emotion dict...')
    content_dict = defaultdict(float)
    label_dict = defaultdict(float)
    other_vec_dict = defaultdict(float)
    with open(datafile, "rb") as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        first_line = True
        for line in csvreader:
            if first_line:
                first_line = False
                continue
            try:
                line.remove('')
            except ValueError:
                None
            uid = line[0]
            uid = uid.strip().lower()
            orig_rev = line[1]
            orig_rev = orig_rev.strip().lower()

            content_dict.setdefault(uid, []).append(orig_rev)

            character_value = []
            for value in line[7:12]:
                character_value.append(1 if value == 'y' else 0)
            label_dict[uid] = character_value

            other_list = line[2:7]
            other_list += line[13:]
            other_vec_dict[uid] = other_list

    print(len(label_dict), len(content_dict), len(other_vec_dict))
    return content_dict, label_dict, other_vec_dict


def load_content(content_dict={}):
    content_lines = []
    for key, values in content_dict.items():
        content_lines.append("".join(values))
    print(len(content_lines))
    return content_lines


def load_label(label_dict={}):
    labels = []
    for key, values in label_dict.items():
        label = []
        for v in values:
            label.append(int(v))
        labels.append(label)
    print(len(labels))
    return labels


def load_other(other_vec_dict={}):
    other_vec = []
    for key, values in other_vec_dict.items():
        other = []
        for v in values:
            other.append(float(v))
        other_vec.append(other)
    print(len(other_vec))
    return other_vec


def write_npy(path, X):
    print(np.array(X))
    np.save(path, np.array(X))


from crawler import Crawler

if __name__ == '__main__':
    content_dict, label_dict, other_vec_dict = readcsv("./fb_data/mypersonality_final.csv")
    content_lines = load_content(content_dict)
    cr = Crawler()
    cr.textmind_action(content_lines)
