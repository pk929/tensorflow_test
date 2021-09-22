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

from common.log import Log
from common.utils import removeFileIfExists
from common.utils import replaceMutiSpace

log = Log()

fileter_punctuation = (string.punctuation + punctuation) \
    .replace('?', '') \
    .replace('？', '')


def segment(text):
    new_text = text
    word_list = jieba.cut(new_text)
    new_text = [word.lower() for word in word_list if word != ' ']
    new_text = " ".join(new_text)
    return new_text


def cleanText(text):
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


word_limit_size = 50

source_data_folder = '/python/TalkRobot/test/02_chinese_corpus/data/corpus_6_4000'

labeled_train_data_csv = os.path.join(basePath, 'data/labeledTrainData.csv')

removeFileIfExists(labeled_train_data_csv)

label_id_dict = {'Auto': 0, 'Culture': 1, 'Economy': 2, 'Medicine': 3, 'Military': 4, 'Sports': 5}

id_label_dict = {v: k for k, v in label_id_dict.items()}

with open(labeled_train_data_csv, 'w', encoding='utf-8', newline='') as csv_file:
    labeled_train_data_writer = csv.writer(csv_file)

    line_data = ['content', 'label_id', 'label']
    labeled_train_data_writer.writerow(line_data)

    for root, dirs, allFiles in os.walk(source_data_folder):
        cnt_files = len(allFiles)
        log.info('一共有{}个文件'.format(cnt_files))

        line_data = None
        content = None
        content_list = None
        label = None
        label_id = None

        wipFileNum = 0
        for file in allFiles:
            log.debug('进度：{}/{}'.format(wipFileNum, cnt_files))

            full_file_path = os.path.join(root, file)
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
            content = replaceMutiSpace(content)

            content = cleanText(content)

            content = content[0:word_limit_size * 4]
            content = segment(content)
            content = ' '.join(content.split(' ')[0:word_limit_size])

            line_data = [content, label_id, label]
            labeled_train_data_writer.writerow(line_data)

            wipFileNum = wipFileNum + 1

log.info('训练数据整理完毕')
