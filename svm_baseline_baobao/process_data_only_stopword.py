# coding=utf-8
"""文本中提取停用词"""
import codecs

__author__ = 'gu'


def write_file(filename, lines):
    f = codecs.open(filename, 'w+')
    query = "".join(lines)
    f.write(query)
    f.close()


def load_data(doc):
    list_total = []
    f = codecs.open(doc)
    lines = f.readlines()
    stop_words = load_stopword()
    for line in lines:
        line = line.replace('\n', '').split()
        user_name = line[0]
        tags = line[1:6]
        query = line[6:]
        only_stop_word = []
        for q in query:
            if q in stop_words:
                only_stop_word.append(q)
        query1 = " ".join(only_stop_word).strip()
        wirte_str = user_name + " " + " ".join(tags).strip() + " " + query1 + "\n"
        list_total.append(wirte_str)
    print(len(list_total))
    print "data have read "
    return list_total


def load_stopword():
    """
    加载停用词语
    :param stopworddoc:
    :return:
    """
    stop_word = []
    # return stop_word
    with open('EN_Stopword.txt') as f:
        lines = f.readlines()
        for line in lines:
            word = line.replace('\n', '')
            if word != '':
                stop_word.append(word)
    with open('ENstopwords.txt') as f:
        lines = f.readlines()
        for line in lines:
            word = line.replace('\n', '')
            if word != '':
                stop_word.append(word)

    return list(set(stop_word))


if __name__ == '__main__':
    lines = load_data("/home/gu/PycharmProjects/tensorflow_demo/essay_data/vocab1_train.txt")
    write_file("/home/gu/PycharmProjects/tensorflow_demo/essay_data/vocab1_train_stopword.txt", lines)
