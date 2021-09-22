import os
import sys

basePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(basePath)

import csv
import re
import jieba
from bs4 import BeautifulSoup
from zhon.hanzi import punctuation as punctuation
import string
import random



from short_text_statistics.utils import utils
from short_text_statistics import logger
log = logger.general_log

class prepareTrainData:
    def __init__(self,word_limit_size):
        self.word_limit_size = word_limit_size

    def prepareTrainData(self):
        fileter_punctuation = (string.punctuation + punctuation) \
            .replace('?', '') \
            .replace('？', '')
        # word_limit_size = 50
        """资源文件"""
        source_data_folder = '/python/TalkRobot/test/02_chinese_corpus/data/corpus_6_4000'
        """训练数据文件"""
        labeled_train_data_csv = os.path.join(basePath, 'data/labeledTrainData.csv')
        utils.removeFileIfExists(labeled_train_data_csv)

        """测试数据的路径文件"""
        test_data_path_csv = os.path.join(basePath, 'data/testDataPath.csv')
        utils.removeFileIfExists(test_data_path_csv)


        label_id_dict = {'Auto': 0, 'Culture': 1, 'Economy': 2, 'Medicine': 3, 'Military': 4, 'Sports': 5}

        id_label_dict = {v: k for k, v in label_id_dict.items()}
        with open(test_data_path_csv, 'w', encoding='utf-8', newline='') as test_csv_file:
            test_data_path_writer = csv.writer(test_csv_file)
            with open(labeled_train_data_csv, 'w', encoding='utf-8', newline='') as csv_file:
                labeled_train_data_writer = csv.writer(csv_file)

                line_data = ['content', 'label_id', 'label']
                labeled_train_data_writer.writerow(line_data)

                for root, dirs, allFiles in os.walk(source_data_folder):
                    cnt_files = len(allFiles)
                    log.info('一共有{}个文件'.format(cnt_files))

                    """生成10%的随机数list"""
                    cut_len = int(cnt_files * 0.1)
                    test_file_list = random.sample(range(1, cnt_files), cut_len)



                    line_data = None
                    content = None
                    content_list = None
                    label = None
                    label_id = None

                    wipFileNum = 0
                    for file in allFiles:
                        log.debug('进度：{}/{}'.format(wipFileNum, cnt_files))

                        # 文件路径
                        full_file_path = os.path.join(root, file)

                        """
                            随机抽取10%的数据用于测试，不进行训练
                        """
                        if wipFileNum in test_file_list:
                            """写入目录"""
                            test_data_path_writer.writerow([full_file_path])
                        else:
                            file_name = file
                            label = file.split('_')[0]
                            label_id = label_id_dict[label]

                            with open(full_file_path, 'r', encoding='utf-8') as f:
                                content_list = f.readlines()

                            content = ''
                            for content_line in content_list:
                                if not (content_line == '\n' or content_line.strip() == ''):
                                    content = content + content_line.replace('\n', '')
                                else:
                                    content = content + ' '
                            content = utils.replaceMutiSpace(content)

                            content = self.cleanText(content)

                            content = content[0:self.word_limit_size * 4]
                            content = self.segment(content)
                            content = ' '.join(content.split(' ')[0:self.word_limit_size])

                            line_data = [content, label_id, label]
                            labeled_train_data_writer.writerow(line_data)


                        wipFileNum = wipFileNum + 1


        log.info('训练数据整理完毕')

    """分词，小写，空格分隔"""
    def segment(self, text):
        stopwords_path = 'D:/python_pycharm_workspace/jieba_test/txt/test_stop.txt'
        stopwords_path = os.path.join(basePath, 'data/stopwords.txt')

        new_text = text

        """jieba分词"""
        word_list = " ".join(jieba.cut(new_text))

        """去除暂停词"""
        mywordlist = utils.removeStopWords(word_list, stopwords_path)

        # 得到分词以空格分隔，将字母转换为小写字母
        new_text = [word.lower() for word in mywordlist if word != ' ']
        new_text = " ".join(new_text)
        return new_text

    """清洗文本"""
    def cleanText(self, text):
        beau = BeautifulSoup(text, "html.parser")
        new_text = beau.get_text()

        pattern = re.compile(r'({53c_min#)(.*)(#})')
        new_text = pattern.sub(r'', new_text)

        pattern = re.compile(r'(\[img\])(.*)(\[\/img\])')
        new_text = pattern.sub(r'', new_text)

        new_text = new_text.replace('?', '？')
        new_text = new_text.replace('!', '！')

        global fileter_punctuation
        new_text = re.sub("[{}]+".format(punctuation), " ", new_text)
        return new_text


