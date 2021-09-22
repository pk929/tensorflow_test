import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import logging
from logging import handlers
import time
import datetime


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, logPath, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):

        # log_file = ''
        # try:
        #     config_path = rootPath + "/config.ini"
        #     cf = configparser.ConfigParser()
        #     cf.read(config_path)
        #     log_file = cf.get("Logger-path", "log_file")
        # except Exception as e:
        #     print(e)

        # log_date = time.strftime("%Y%m%d")
        # log_datetime = time.strftime("%Y%m%d_%H%M%S")
        # log_path = log_file + '/log/' + log_date

        log_month = time.strftime("%Y%m")
        log_date = time.strftime("%Y%m%d")
        log_datetime = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(logPath, log_month)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        filename = os.path.join(log_path, 'log-'+ log_date +'.log')
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)

"""普通日志"""
_general_log_path = os.path.join(curPath, 'log', 'generalLog')
"""统计日志"""
_statistical_log_path = os.path.join(curPath, 'log', 'statisticalLog')

general_log = Logger(_general_log_path).logger
statistical_log = Logger(_statistical_log_path).logger

general_log.info('\n\n\n\t\t新启动运行T%s\n\n\n' %(datetime.datetime.now()))
statistical_log.info('\n\n\n\t\t新启动运行T%s\n\n\n'%(datetime.datetime.now()))
