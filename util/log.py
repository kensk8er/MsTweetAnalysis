"""
log module

Utility functions related to log are defined here.
"""
import logging
import os
from util.string import strip_suffix

__author__ = 'kensk8er'


def configure_log(path):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    dir_name, file_name = os.path.split(path)
    file_name = strip_suffix(string=file_name, suffix='.py')
    log = logging.getLogger(file_name)
    file_handler = logging.FileHandler('log/' + file_name + '.log')
    file_handler.level = logging.INFO
    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s')
    file_handler.formatter = formatter
    log.addHandler(file_handler)
    return log
