# coding:utf-8

"""
爬去文心系统的数据，提取特征
"""

import urllib2
import urllib
import cookielib
import json

import numpy as np

headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:35.0) Gecko/20100101 Firefox/35.0'}


class Crawler:
    def __init__(self):
        self.cj = cookielib.LWPCookieJar()
        self.cookie_processor = urllib2.HTTPCookieProcessor(self.cj)
        self.opener = urllib2.build_opener(self.cookie_processor, urllib2.HTTPHandler)
        urllib2.install_opener(self.opener)

    def doPost(self, text):
        print "正在请求文心..."
        PostData = {
            "str": text
        }
        PostData = urllib.urlencode(PostData)
        request = urllib2.Request('http://ccpl.psych.ac.cn/textmind/analysis', PostData, headers)
        response = urllib2.urlopen(request)
        text = response.read()
        print text
        return text

    def parse_textmind_feature(self, json_str):
        feature_list = []
        json_dict = json.loads(json_str)
        print(json_dict)
        if json_dict['status'] == 'success':
            result_list = json_dict['result']
            for elem in result_list:
                name = elem['name']
                value = elem['value']
                feature_list.append(value)
        else:
            raise ValueError('文心系统分析返回数据异常')
        return feature_list

    def save_arr(self, filename, X_sp):
        """
        特征向量保存
        """
        np.save(filename, X_sp)
        print "*****************write done over *****************"

    def textmind_action(self, train_lines, test_lines):
        """
        输入文本[] 保存特征
        :param train_lines:
        :param test_lines:
        :return:
        """
        X_train = self.get_input_output(train_lines)
        X_test = self.get_input_output(test_lines)

        textmind_train_vec_dm = "textmind_train_vec.npy"
        textmind_train_label_dm = "textmind_train_label.npy"
        textmind_test_vec_dm = "textmind_test_vec.npy"
        textmind_test_label_dm = "textmind_test_label.npy"

        self.save_arr(textmind_train_vec_dm, np.array(X_train))
        self.save_arr(textmind_test_vec_dm, np.array(X_test))

    def get_input_output(self, lines):
        """
        输入文本的lines 返回每行对应的文心特征 和 对应的标签
        :param lines:
        :return:
        """
        list_input_feature = []
        for t_line in lines:
            json_str = self.doPost(t_line)
            feature_list = self.parse_textmind_feature(json_str)
            list_input_feature.append(feature_list)
        return list_input_feature
