"""
Preprocessing both user and job corpus.
"""
import logging
from gensim.corpora.dictionary import Dictionary
from gensim.utils import lemmatize, revdict
import re
from module.text.preprocessing import convert_compound, clean_text
from module.text.stopword import user_stopwords, job_stopwords
from util.input import unpickle
from util.output import enpickle

__author__ = 'kensk8er'


def preprocess_user_corpora(corpora, stopwords, allowed_pos, max_doc=float('inf'), no_above=0.5, no_below=5,
                            keep_n=None):
    """


    :rtype : gensim.corpora.dictionary.Dictionary
    :param corpora: 
    :param stopwords: 
    :param allowed_pos: 
    :param max_doc: 
    :return: 
    """
    logging.info('Lemmatizing the corpora...')
    count = 0
    corpus_num = len(corpora)
    processed_corpora = []
    corpus_id2user_id = []

    for index, corpus in corpora.items():
        count += 1
        if count > max_doc:
            break
        if corpus is None:  # skip if corpus is None
            continue

        print '\r', count, '/', corpus_num,
        cleaned_corpus = clean_text(corpus)  # delete irrelevant characters
        corpus = []
        tokens = lemmatize(content=cleaned_corpus, allowed_tags=allowed_pos)
        for token in tokens:
            word, pos = token.split('/')
            corpus.append(word)

        # convert compound word into one token
        corpus = convert_compound(corpus)

        # filter stop words, long words, and non-english words
        corpus = [w for w in corpus if not w in stopwords and 2 <= len(w) <= 15 and w.islower()]
        processed_corpora.append(corpus)
        corpus_id2user_id.append(index)

    print '\n'

    enpickle(corpus_id2user_id, 'data/dictionary/corpus_id2user_id.pkl')
    logging.info('Creating dictionary and corpus...')
    dictionary = Dictionary(processed_corpora)
    dictionary.corpus_id2user_id = corpus_id2user_id

    logging.info('Filtering unimportant terms...')
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()

    logging.info('Generating corpus...')
    dictionary.corpus = [dictionary.doc2bow(corpus) for corpus in processed_corpora]
    dictionary.id2token = revdict(dictionary.token2id)
    dictionary.save('data/dictionary/user_' + allowed_pos.pattern + '.dict')

    return dictionary


# TODO: merge with preprocess_user_corpora nicely
def preprocess_job_corpora(corpora, stopwords, allowed_pos, max_doc=float('inf'), no_above=0.5, no_below=5,
                           keep_n=None):
    """


    :rtype : gensim.corpora.dictionary.Dictionary
    :param corpora:
    :param stopwords:
    :param allowed_pos:
    :param max_doc:
    :return:
    """
    logging.info('Lemmatizing the corpora...')
    count = 0
    corpus_num = len(corpora)
    processed_corpora = []
    corpus_id2job_id = []

    for index, corpus in corpora.items():
        count += 1
        if count > max_doc:
            break
        if corpus is None:  # skip if corpus is None
            continue

        print '\r', count, '/', corpus_num,
        cleaned_corpus = clean_text(corpus)  # delete irrelevant characters
        corpus = []
        tokens = lemmatize(content=cleaned_corpus, allowed_tags=allowed_pos)
        for token in tokens:
            word, pos = token.split('/')
            corpus.append(word)

        # convert compound word into one token
        corpus = convert_compound(corpus)

        # filter stop words, long words, and non-english words
        corpus = [w for w in corpus if (not w in stopwords) and 2 <= len(w) <= 15 and w.islower()]
        processed_corpora.append(corpus)
        corpus_id2job_id.append(index)

    print '\n'
    enpickle(corpus_id2job_id, 'data/dictionary/corpus_id2job_id.pkl')

    logging.info('Creating dictionary and corpus...')
    dictionary = Dictionary(processed_corpora)
    dictionary.corpus_id2job_id = corpus_id2job_id

    logging.info('Filtering unimportant terms...')
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()

    logging.info('Generating corpus...')
    dictionary.corpus = [dictionary.doc2bow(corpus) for corpus in processed_corpora]
    dictionary.id2token = revdict(dictionary.token2id)
    dictionary.save('data/dictionary/job_' + allowed_pos.pattern + '.dict')

    return dictionary


if __name__ == '__main__':
    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    logging.info('Loading user corpora...')
    user_corpora = unpickle('data/rawcorpus/user_corpora.pkl')

    logging.info('Preprocessing user corpora...')
    user_dict = preprocess_user_corpora(corpora=user_corpora, stopwords=user_stopwords, allowed_pos=re.compile('(NN)'))

    logging.info('Loading job corpora...')
    job_corpora = unpickle('data/rawcorpus/job_corpora.pkl')

    logging.info('Preprocessing job corpora...')
    job_dict = preprocess_job_corpora(corpora=job_corpora, stopwords=job_stopwords, allowed_pos=re.compile('(NN)'))

    # TODO: merge function for generating wiki corpus