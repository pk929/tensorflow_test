import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import re
import shutil
import jieba


from short_text_statistics import logger
from short_text_statistics import logger
log = logger.general_log


class Utils:
    """去除暂停词"""

    def removeStopWords(self, word_list, stopwords_path):
        f_stop_text = ''
        if os.path.exists(stopwords_path):
            f_stop_text = open(stopwords_path, encoding='utf-8').read()
        f_stop_seg_list = f_stop_text.split('\n')
        mywordlist = []
        for myword in word_list.split(' '):
            if not (myword.strip() in f_stop_seg_list):
                mywordlist.append(myword)
        return mywordlist



    def del_file(seif, filepath):
        """
        删除某一目录下的所有文件或文件夹
        :param filepath: 路径
        :return:
        """
        if os.path.exists(filepath):
            log.info('删除文件夹内容：%s' %filepath)
            del_list = os.listdir(filepath)
            for f in del_list:
                file_path = os.path.join(filepath, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    def removeFileIfExists(self,filePath):
        """
        如果文件已经存在，就删除文件
        :param filePath:
        :return:
        """
        if os.path.isfile(filePath):
            os.remove(filePath)
            log.info('已经删除存在的文件{}'.format(filePath))


    def replaceMutiSpace(self,str):
        """
        多个空格替换成单个空格
        :param str:
        :return:
        """
        str = re.sub(' +', ' ', str)
        return str

    def getFolderSize(self, folderPath):
        size = 0
        log.info('获取文件夹大小：%s' % folderPath)
        if os.path.exists(folderPath):
            for root, dirs, files in os.walk(folderPath):
                size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
        return size




log = logger.general_log
utils = Utils()