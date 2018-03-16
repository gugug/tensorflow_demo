# coding=utf-8
import multiprocessing
from gensim.models import word2vec
from nltk.corpus import stopwords

__author__ = 'gu'


class MySentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname, 'r'):
            yield line.split()


DataDir = '/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/w2v/'

ModelDir = "./ipynb_garbage_files/"
MIN_COUNT = 4
CPU_NUM = multiprocessing.cpu_count()  # 需要预先安装 Cython 以支持并行
VEC_SIZE = 20
CONTEXT_WINDOW = 5  # 提取目标词上下文距离最长5个词
f_input = "bioCorpus_5000.txt"
model_output = "test_w2v_model"


# ================================2.2. Word2vec 训练===========================

# 模型训练函数 指定输入的文件路径和待输出的模型路径
def w2vTrain(f_input, model_output):
    sentences = MySentences(DataDir + f_input)
    w2v_model = word2vec.Word2Vec(sentences,

                                  min_count=MIN_COUNT,
                                  # min_count: 对于词频 < min_count 的单词，将舍弃（其实最合适的方法是用 UNK 符号代替，即所谓的『未登录词』，这里我们简化起见，认为此类低频词不重要，直接抛弃）

                                  workers=CPU_NUM,
                                  # 可以并行执行的核心数
                                  size=VEC_SIZE,
                                  # 词向量的维度，即神经网络隐层节点数
                                  window=CONTEXT_WINDOW
                                  # 目标词汇的上下文单词距目标词的最长距离，很好理解，比如 CBOW 模型是用一个词的上下文预测这个词，那这个上下文总得有个限制，如果取得太多，距离目标词太远，有些词就没啥意义了，而如果取得太少，又信息不足，所以 window 就是上下文的一个最长距离
                                  )

    w2v_model.save(ModelDir + model_output)

# 训练
w2vTrain(f_input, model_output)


# ============================2.3. 查看结果================

# # 加载模型
# w2v_model = word2vec.Word2Vec.load(ModelDir + model_output)
#
# # 找一些现有词的相似词
# print w2v_model.most_similar('body')  # 结果一般
#
# print w2v_model.most_similar('heart')  # 结果太差

"""
模型调优
混入的这些奇怪的东西，在 NLP 里面我们叫『停止词』，也就是像常见代词、介词之类，造成这种结果的原因我认为有二

参数设置不佳，比如 vec_size 设置的太小，导致这 20 个维度不足以 capture单词间不同的信息，所以我们需要继续调整超参数
数据集较小，因此停止词占据了太多信息量
"""
StopWords = stopwords.words('english')

# 重新训练# 模型训练函数

def w2vTrain_removeStopWords(f_input, model_output):
    sentences = list(MySentences(DataDir + f_input))
    for idx, sentence in enumerate(sentences):
        sentence = [w for w in sentence if w not in StopWords]
        sentences[idx] = sentence

    w2v_model = word2vec.Word2Vec(sentences, min_count=MIN_COUNT,
                                  workers=CPU_NUM, size=VEC_SIZE)
    w2v_model.save(ModelDir + model_output)

# 训练
w2vTrain_removeStopWords(f_input, model_output)

# 查看
w2v_model = word2vec.Word2Vec.load(ModelDir + model_output)
print w2v_model.most_similar('body')
print w2v_model.most_similar('heart')

print w2v_model.wv['body']
