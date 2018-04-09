# coding=utf-8
# ==============================================================================

"""Functions for reading data."""

import os

import numpy
import numpy as np
from tensorflow.python.platform import gfile

CHARACTER_LABELS = {'适应性': 0, '社交性': 1, '开放性': 2, '利他性': 3, '道德感': 4}
train_base_path = '/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/character_data/训练集200个/训练集'
test_base_path = '/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/character_data/评估集40个/评估集'
all_label_path = "/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/character_data/all_data/all_user_character_map.txt"
vec_path = '/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/character_data/vec'
npy_train_label = vec_path + "/train_label.npy"
npy_test_label = vec_path + "/test_label.npy"
npy_train_vec = vec_path + "/train_vec.npy"
npy_test_vec = vec_path + "/test_vec.npy"
npy_textmind_train = vec_path + "/textmind_train_vec.npy"
npy_textmind_test = vec_path + "/textmind_test_vec.npy"


def combine_blog_to_one(blog_dir):
    """combine the blog content under blog_dir into one txt"""
    files = gfile.ListDirectory(blog_dir)
    print(len(files))
    for f in files:
        one_dir = os.path.join(blog_dir, f)
        blog_files = gfile.ListDirectory(one_dir)
        content_file = open(os.path.join(one_dir, "all_content.txt"), "w+")
        for bf in blog_files:
            filename = os.path.join(one_dir, bf)
            content = extract_content(filename)
            content_file.write(content)
        content_file.close()


def extract_content(filename):
    """
    return second line in content in filename
    :param filename:
    :return:
    """
    with gfile.Open(filename, 'rb') as gf:
        lines = gf.readlines()
        return lines[1]


def extract_faw(train_dir):
    """
    提取特征数据
    提取train下info.txt的性别，粉丝，关注，微博
    :param train_dir:
    :return:
    """
    files = gfile.ListDirectory(train_dir)
    train_data_file = open(
        os.path.join("/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/character_data",
                     "vec/test_vec.txt"), "w+")
    for userid in files:
        infofilepath = os.path.join(train_dir, userid, 'info.txt')
        faw_list = extract_info_faw(infofilepath)
        for faw in faw_list:
            train_data_file.write(str(faw) + " ")
        train_data_file.write('\n')
    train_data_file.close()


def extract_info_faw(filename):
    with gfile.Open(filename, 'rb') as gf:
        lines = gf.readlines()
        sex = lines[2].replace("\n", '').split(":")[1]
        if sex == '男':
            sex = 1
        elif sex == '女':
            sex = -1
        else:
            sex = 0
        fans = lines[-1].replace("\n", '').split(":")[1]
        atten = lines[-2].replace("\n", '').split(":")[1]
        weibo = lines[-3].replace("\n", '').split(":")[1]
        return [sex, int(fans), int(atten), int(weibo)]


def extract_labels(filename):
    """
    提取性格标签
    :param filename:
    :return:
    """
    print('Extracting', filename)
    with gfile.Open(filename, 'rb') as f:
        lines = f.readlines()
        ids = []
        labels = []
        print('labels size are ', len(lines))
        for line in lines:
            line = line.replace('\n', '')
            id_character = line.split('\t\t\t\t\t\t')
            ids.append(id_character[0])
            labels.append(CHARACTER_LABELS.get(id_character[1].split('、')[0]))
        write_labels(filename + '.txt', ids, labels)  # id对应标签
        return numpy.array(ids), numpy.array(labels)


def write_labels(filename, ids, labels):
    with gfile.Open(filename, 'w+') as f:
        for idx in range(len(ids)):
            f.write(ids[idx] + " " + str(labels[idx]))
            f.write('\n')


def parallel_label(train_dir, alllabelfilename):
    """parallel the data and labels"""
    id_label_dict = {}
    with gfile.Open(alllabelfilename, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            id_label = line.split(" ")
            id_label_dict.setdefault(id_label[0], id_label[1])
    train_txt = open(
        os.path.join("/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/character_data",
                     "vec/test_label.txt"),
        'w+')
    userid_list = gfile.ListDirectory(train_dir)
    for id in userid_list:
        print(id)
        print(id_label_dict.get(id))
        train_txt.write(str(id_label_dict.get(id)))
        train_txt.write('\n')
    train_txt.close()


def load_content(train_dir):
    """
    提取特征数据
    提取train下all_content.txt内容
    :param train_dir:
    :return:
    """
    content_list = []
    userid_list = []
    files = gfile.ListDirectory(train_dir)
    for userid in files:
        userid_list.append(userid)
        allcontent_filepath = os.path.join(train_dir, userid, 'all_content.txt')
        with gfile.Open(allcontent_filepath, 'rb') as gf:
            content = gf.read()
            content_list.append(content)
    return userid_list, content_list


def load_faw(filename):
    """load data"""
    xs = []
    with gfile.Open(filename, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 0:
                line = line.replace("\n", "")
                id_label = line.split(" ")
                xs.append([float(id_label[0]), float(id_label[1]), float(id_label[2]), float(id_label[3])])
    return xs


def load_label(filename):
    """label"""
    ys = []
    with gfile.Open(filename, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 0:
                line = line.replace("\n", "")
                ys.append(line)
    return ys


def write_npy(path, X):
    print(np.array(X))
    np.save(path, np.array(X))


def load_npy(path):
    X = np.load(path)
    return X


if __name__ == '__main__':
    pass
