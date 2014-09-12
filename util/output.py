"""
output module

Utility functions related to output are defined here.
"""
from util.input import unpickle

__author__ = 'kensk8er'


def enpickle(data, file):
    import cPickle

    fo = open(file, 'w')
    cPickle.dump(data, fo, protocol=2)
    fo.close()
