"""
corpus module

Utility functions related to corpus are defined here.
"""
import pyximport
pyximport.install()
import _corpus

__author__ = 'kensk8er'


def corpus2vector(corpora, num_terms):
    return _corpus.corpus2vector(corpora=corpora, num_terms=num_terms)  # use cython function


def filter_term(corpora, valid_term_ids):
    return _corpus.filter_term(corpora=corpora, valid_term_ids=valid_term_ids)  # use cython function
