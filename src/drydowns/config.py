import os
import copy
import configparser

import logging
from logging.config import dictConfig


logging_config = dict(
    version=1,
    formatters={
        'f': {'format':
              '%(relativeCreated)6d %(levelname)-8s %(name)-12s %(message)s'}
    },
    handlers={
        'ch': {'class': 'logging.StreamHandler',
               'formatter': 'f',
               'level': logging.DEBUG},
        'fh': {'class': 'logging.FileHandler',
               'formatter': 'f',
               'filename': 'error.log',
               'mode': 'w',
               'encoding': 'utf-8',
               'level': logging.DEBUG}
    },
    root={
        'handlers': ['ch', 'fh'],
        'level': logging.DEBUG,
    },
)


dictConfig(logging_config)
logger = logging.getLogger()


config_type = os.getenv('CONFIG', 'default')


class Config:
    def __init__(self, ini_file=None):
        if ini_file:
            self.config = Config.create_config(ini_file=ini_file)
            self.config_file = ini_file
        self.status = True
        self.logger = logger
        # self.logger.setLevel(logging.NOTSET)
        self.logger.setLevel(logging.DEBUG)
            

    def __repr__(self):
        class_name = type(self).__name__
        repr = '{}('
        repr += 'ini_file="{}"'
        return '{}(ini_file="{}")'.format(class_name, self.config_file)
    
    @staticmethod
    def create_config(ini_file=None):
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        config.read(ini_file)

        # for src in config.sections():
        #     # Get directory path
        #     path = config.get(src, 'dir')
        #     # Expand path (allows input of ~ in config.ini)
        #     config.set(src, 'dir', os.path.expanduser(path))

        return config