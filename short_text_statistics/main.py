import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


import time
from short_text_statistics import logger
from short_text_statistics import prepareTrainData
from short_text_statistics import trainModel
from short_text_statistics import predictions
from short_text_statistics.utils import utils

statistical_log = logger.statistical_log
general_log = logger.general_log


class WordVectorParam:
    def __init__(self):
        self.sentences = None # 供训练的句子，可以使用简单的列表，但是对于大语料库，建议直接从磁盘 / 网络流迭代传输句
        self.corpus_file = None # LineSentence格式的语料库文件路径， [["cat", "say", "meow"], ["dog", "say", "woof"]]
        self.size = 100 # 词向量的维度，默认100，取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度
        self.window = 5 # 词向量上下文最大距离, 默认值为5, window越大，则和某一词较远的词也会产生上下文关系。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5, 10]之间
        self.min_count = 5 # 需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值
        self.sg = 0 # 默认是0。word2vec两个模型的选择，如果是0，则是CBOW模型，是1则是Skip - Gram模型
        self.cbow_mean = 1 #  0 # 使用上下文单词向量的总和;1 # 使用均值，适用于使用CBOW
        self.hs = 0 # 默认是0。word2vec两个解法的选择，如果是0，则是NegativeSampling，是1的话并且负采样个数negative大于0， 则是HierarchicalSoftmax。
        self.negative = 5 # 使用NegativeSampling时负采样的个数，默认是5。推荐在[3, 10]之间
        self.ns_exponent = 0.75 #  负采样分布指数。1.0样本值与频率成正比，0.0样本所有单词均等，负值更多地采样低频词
        self.iter = 5 # 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值
        self.alpha = 0.025 # 在随机梯度下降法中迭代的初始步长。默认是0.025。
        self.min_alpha = 0.0001 # 随着训练的进行，学习率线性下降到min_alpha。 对于大语料，需要对alpha, min_alpha, iter一起调参
        self.hashfxn = hash  # 哈希函数用于随机初始化权重，以提高训练的可重复性
        self.null_word = 0 #
        self.trim_rule = None # 词汇修剪规则，指定某些词语是否应保留在词汇表中，修剪掉或使用默认值处理
        self.sorted_vocab = 1 # 如果为1，则在分配单词索引前按降序对词汇表进行排序
        self.workers = 3 #  训练模型时使用的线程数, worker参数只有在安装了Cython后才有效.没有Cython的话, 只能使用单核
        self.batch_words = 10000 # 每一个batch传递给线程单词的数量
        self.max_final_vocab = None #自动选择匹配的min_count将词汇限制为目标词汇大小
        self.max_vocab_size = None #  词汇构建期间RAM的限制;如果有更多的独特单词，则修剪不常见的单词。 每1000万个类型的字需要大约1GB的RAM
        self.seed = 1 #  随机数发生器种子
        self.sample = 1e-3 # 高频词随机下采样的配置阈值，范围是(0, 1e-5)
        self.compute_loss = False # 如果为True，则计算并存储可使用get_latest_training_loss()检索的损失值
        self.callbacks = () #  在训练中特定阶段执行回调序列

"""整理训练数据"""
def _prepareTrain(word_limit_size):
    statistical_log.info('整理训练数据')
    starttime = time.time()
    prepareTrainData.prepareTrainData(word_limit_size).prepareTrainData()
    endtime = time.time()
    use = endtime - starttime
    statistical_log.info('word_limit_size=%s' % (word_limit_size))
    statistical_log.info('整理训练数据时间:%s毫秒;%s秒' % (int(round((use) * 1000)), int(use)))  # 毫秒
    statistical_log.info('\n\n')

"""生成词向量文件"""
def _wordVector(wordVectorParam):
    """
        (sentences=None, corpus_file=None, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(),
                 max_final_vocab=None)
                 
    """

    statistical_log.info('生成词向量文件')
    statistical_log.info('生成词向量参数WordVectorParam:%s' % (wordVectorParam.__dict__))
    starttime = time.time()
    trainModel.WordVector(wordVectorParam).generateWord2VectorFile()
    endtime = time.time()
    use = endtime - starttime
    statistical_log.info('生成词向量文件时间:%s毫秒;%s秒' % (int(round((use) * 1000)), int(use)))  # 毫秒
    statistical_log.info('\n\n')

"""训练模型"""
def _trainingModel():
    saved_model_file = os.path.join(curPath, 'data/model')
    """之前删除模型文件"""

    utils.del_file(saved_model_file)
    statistical_log.info('训练模型')
    starttime = time.time()
    trainModel.TrainingModel().trainingModel()
    endtime = time.time()
    use = endtime - starttime
    """获取文件夹下模型文件大小"""
    file_size = utils.getFolderSize(saved_model_file)
    statistical_log.info('训练模型时间:%s毫秒;%s秒;模型文件大小:%s' % (int(round((use) * 1000)), int(use), file_size))  # 毫秒
    statistical_log.info('\n\n')

"""准确率统计"""
def _predictions(sequenceLength):
    statistical_log.info('准确率统计')
    statistical_log.info('sequenceLength=%s' % (sequenceLength))
    starttime = time.time()
    pass_count, all_count = predictions.Predictions(sequenceLength).predictions()
    endtime = time.time()
    use = endtime - starttime
    statistical_log.info('成功数：{} 总数：{} 准确率:{}'.format(pass_count, all_count,
                                                      format(float(pass_count) / float(all_count), '.4f')))
    statistical_log.info('准确率统计时间:%s毫秒;%s秒' % (int(round((use) * 1000)), int(use)))  # 毫秒
    statistical_log.info('\n\n')



def aaa(wordVectorParam):
    statistical_log.info('--------------------------------------------')

    """生成词向量文件"""
    try:
        _wordVector(wordVectorParam)
    except Exception as e:
        statistical_log.info('生成词向量文件error:%s' %(e))

    """训练模型"""
    try:
        _trainingModel()
    except Exception as e:
        statistical_log.info('训练模型error:%s' %(e))


    """准确率统计"""
    try:
        _predictions(200)
    except Exception as e:
        statistical_log.info('准确率统计error:%s' %(e))


if __name__ == '__main__':

    """词长度"""
    word_length = 50


    word_limit_size = word_length #整理训练数据时字数限制大小
    sequenceLength = 200
    # sequenceLength = word_length


    """
            (sentences=None, corpus_file=None, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(),
                 max_final_vocab=None)
                 
    
    
    """

    """整理训练数据"""
    try:
        _prepareTrain(word_limit_size)
    except Exception as e:
        statistical_log.info('整理训练数据error:%s' %(e))

    # 配置参数

    """
        window=5: 词向量上下文最大距离,默认值为5,window越大，则和某一词较远的词也会产生上下文关系。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10]之间
    """
    for window in range(2, 11):
        wordVectorParam = WordVectorParam()
        wordVectorParam.window = window
        # 生成文件、模型训练、准确率
        aaa(wordVectorParam)
    """
        min_count=5: 需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值
    """
    for min_count in range(3, 8):
        wordVectorParam = WordVectorParam()
        wordVectorParam.min_count = min_count
        # 生成文件、模型训练、准确率
        aaa(wordVectorParam)
    """
        sg=0: 默认是0。word2vec两个模型的选择，如果是0，则是CBOW模型，是1则是Skip-Gram模型
    """
    for sg in range(1, 2):
        wordVectorParam = WordVectorParam()
        wordVectorParam.sg = sg
        # 生成文件、模型训练、准确率
        aaa(wordVectorParam)
    """
        cbow_mean=1:  0: 使用上下文单词向量的总和; 1: 使用均值，适用于使用CBOW
    """
    for cbow_mean in range(0, 1):
        wordVectorParam = WordVectorParam()
        wordVectorParam.cbow_mean = cbow_mean
        # 生成文件、模型训练、准确率
        aaa(wordVectorParam)

        #测一下Skip-Gram模型时cbow_mean=0的情况
        wordVectorParam.sg = 1
        aaa(wordVectorParam)



    """
        hs=0: 默认是0。word2vec两个解法的选择，如果是0，则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。
    """
    for hs in range(1, 2):
        wordVectorParam = WordVectorParam()
        wordVectorParam.hs = hs
        # 生成文件、模型训练、准确率
        aaa(wordVectorParam)
    """
        negative=5: 使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间
    """
    for negative in range(3, 11):
        wordVectorParam = WordVectorParam()
        wordVectorParam.negative = negative
        # 生成文件、模型训练、准确率
        aaa(wordVectorParam)


    # ############


    """
        ns_exponent=0.75:  负采样分布指数。1.0样本值与频率成正比，0.0样本所有单词均等，负值更多地采样低频词
    """
    for ns_exponent in [0, 0.25, 0.5, 0.75, 1]:
        wordVectorParam = WordVectorParam()
        wordVectorParam.ns_exponent = ns_exponent
        # 生成文件、模型训练、准确率
        aaa(wordVectorParam)

    """
        iter = 5: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值
    """
    for iter in range(3, 9):
        wordVectorParam = WordVectorParam()
        wordVectorParam.iter = iter
        # 生成文件、模型训练、准确率
        aaa(wordVectorParam)

    """
        alpha=0.025: 在随机梯度下降法中迭代的初始步长。默认是0.025。
    """
    for alpha in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.05, 0.1, 0.2]:
        wordVectorParam = WordVectorParam()
        wordVectorParam.alpha = alpha
        # 生成文件、模型训练、准确率
        aaa(wordVectorParam)


    """
        min_alpha=0.0001: 随着训练的进行，学习率线性下降到min_alpha。 对于大语料，需要对alpha, min_alpha,iter一起调参
    """
    for min_alpha in [0.00001, 0.00005, 0.0001, 0.00015, 0.0002]:
        wordVectorParam = WordVectorParam()
        wordVectorParam.min_alpha = min_alpha
        # 生成文件、模型训练、准确率
        aaa(wordVectorParam)


