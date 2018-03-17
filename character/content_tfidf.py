# coding=utf-8
"""
文本的tf-idf值

CountVectorizer类会将文本中的词语转换为词频矩阵，例如矩阵中包含一个元素a[i][j]，
它表示j词在i类文本下的词频。它通过fit_transform函数计算各个词语出现的次数，
通过get_feature_names()可获取词袋中所有文本的关键字，通过toarray()可看到词频矩阵的结果。

TfidfTransformer用于统计vectorizer中每个词语的TF-IDF值
"""
import os

__author__ = 'gu'
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import base
import jieba


# # 语料
# corpus = [
#     'This is the first document.',
#     'This is the second second document.',
#     'And the third one.',
#     'Is this the first document?',
# ]


def load_corpus(filename):
    """load corpus"""
    lines = base.load_content(filename)
    corpus = []
    stopwords = stop_words_list('stopwords.txt')  # 这里加载停用词的路径
    for line in lines[:50]:
        corpus.append(seg_sentence(line, stopwords))
    # print(corpus)
    print('\n '.join(corpus))


# 创建停用词list
def stop_words_list(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
    return stopwords


# 对句子进行分词
def seg_sentence(sentence, stopwords):
    sentence_seged = jieba.cut(sentence.strip())
    outstr = ''
    for word in sentence_seged:
        if word.encode('utf-8') not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


def text_tfidf(corpus):
    # 将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer()
    # 计算个词语出现的次数
    X = vectorizer.fit_transform(corpus)
    # 获取词袋中所有文本关键词
    word = vectorizer.get_feature_names()
    print word
    # 查看词频结果
    print X.toarray()
    # 类调用
    transformer = TfidfTransformer()
    # print transformer
    # 将词频矩阵X统计成TF-IDF值
    tfidf = transformer.fit_transform(X)
    # 查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
    print tfidf.toarray()


if __name__ == '__main__':
    base_path = '/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/character_data/'
    # extract_labels(os.path.join(base_path,'user_character_map'))
    load_corpus(os.path.join(base_path, '训练集200个/训练集/陈传管住嘴/all_content.txt'))
