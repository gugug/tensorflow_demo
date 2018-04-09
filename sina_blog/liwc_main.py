# coding=utf-8
__author__ = 'gu'
from character import load_content
from crawler import Crawler

if __name__ == '__main__':
    craw = Crawler()
    train_base_path = '/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/character_data/训练集200个/训练集'
    train_userid_list, train_content_list = load_content(train_base_path)
    test_base_path = '/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/character_data/评估集40个/评估集'
    test_userid_list, test_content_list = load_content(test_base_path)
    print(len(train_userid_list), len(train_content_list))
    print(len(test_userid_list), len(test_content_list))
    craw.textmind_action([], test_content_list)
