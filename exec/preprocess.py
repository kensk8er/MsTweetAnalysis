"""
Preprocessing corpora
"""
import logging
from gensim.corpora.dictionary import Dictionary
from gensim.utils import lemmatize, revdict
import re
from module.text.preprocessing import convert_compound, clean_text
from module.text.stopword import stopwords
from util.input import unpickle

__author__ = 'kensk8er'


def preprocess_corpora(corpora, stopwords, allowed_pos, max_doc=float('inf'), no_above=0.5, no_below=1, keep_n=None):
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
    corpus_id2orig_id = []

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
        corpus_id2orig_id.append(index)

    print '\n'

    logging.info('Creating dictionary and corpus...')
    dictionary = Dictionary(processed_corpora)
    dictionary.corpus_id2orig_id = corpus_id2orig_id

    logging.info('Filtering unimportant terms...')
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()

    logging.info('Generating corpus...')
    dictionary.corpus = [dictionary.doc2bow(corpus) for corpus in processed_corpora]
    dictionary.id2token = revdict(dictionary.token2id)

    return dictionary


if __name__ == '__main__':
    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    logging.info('Loading corpora...')
    tweets_corpora = unpickle('data/processed/tweets.pkl')

    logging.info('Preprocessing corpora...')
    dictionary = preprocess_corpora(corpora=tweets_corpora, stopwords=stopwords, allowed_pos=re.compile('(NN)'))
    dictionary.save('data/dictionary/tweets.dict')