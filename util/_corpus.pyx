"""
Cython file
"""
import numpy as np

__author__ = 'kensk8er'

def corpus2vector(corpora, int num_terms):
    vectors = np.zeros((len(corpora), num_terms))
    cdef int corpus_id
    cdef int tuple_id
    for corpus_id in range(len(corpora)):
        if corpus_id % 100 == 0:
            print '\r', str(corpus_id), '/', str(len(corpora)),

        for tuple_id in range(len(corpora[corpus_id])):
            vectors[corpus_id][corpora[corpus_id][tuple_id][0]] = corpora[corpus_id][tuple_id][1]
    print '\r', str(corpus_id + 1), '/', str(len(corpora))
    return vectors


def filter_term(list corpora, list valid_term_ids):
    return _filter_term(corpora, valid_term_ids)

cdef list _filter_term(list corpora, list valid_term_ids):
    cdef unsigned int corpus_id
    cdef unsigned int term_id
    cdef unsigned int frequency
    cdef unsigned int i
    cdef list new_corpora = [[] for i in range(len(corpora))]
    cdef dict _valid_term_ids = {x:0 for x in valid_term_ids}

    for corpus_id in range(len(corpora)):
        if corpus_id % 100 == 0:
            print '\r', str(corpus_id), '/', str(len(corpora)),

        for tuple_id in range(len(corpora[corpus_id])):
            term_id = corpora[corpus_id][tuple_id][0]

            if _valid_term_ids.has_key(term_id):  # this way is super faster than `if term_id in valid_term_ids`!!
                frequency = corpora[corpus_id][tuple_id][1]
                new_corpora[corpus_id].append((term_id, frequency))

    print '\r',str(corpus_id + 1), '/', str(len(corpora))
    return new_corpora
