"""
Executable file for LDA (Latent Dirichlet Allocation).

When running this file, 'working directory' need to be specified as Project Root (MsTweetAnalysis).
"""
import logging
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
from util.input import unpickle
from util.output import enpickle

__author__ = 'kensk8er'


def perform_lda(dictionary, corpus, num_topics, wiki_path=None, passes=1, iterations=50, chunksize=200):
    """


    :param dictionary:
    :param corpus:
    :param wiki_path:
    :param num_topics:
    :param passes:
    :param iterations:
    :param chunksize:
    :return:
    """
    if wiki_path is not None:
        logging.info('Generating wiki corpus...')
        wikis = unpickle(wiki_path)
        wiki_corpus = [dictionary.doc2bow(wiki) for wiki in wikis]

        logging.info('Combining original corpus and wiki corpus...')
        corpus = corpus + wiki_corpus  # wiki_corpus is merged after the original corpus

    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes,
                         iterations=iterations, alpha='auto', chunksize=chunksize)
    doc_vectors = lda_model.inference(corpus)[0]
    doc_vectors = doc_vectors / doc_vectors.sum(axis=1).reshape(doc_vectors.shape[0], 1)

    return lda_model, doc_vectors


def do_lda(num_topics, passes, iterations, chunksize, tfidf, wiki_path=None):
    model_name = 'tweets_'

    logging.info('Loading user dictionary...')
    dictionary = corpora.Dictionary.load('data/dictionary/tweets.dict')
    corpus = dictionary.corpus

    if tfidf is True:
        logging.info('Computing TF-IDF...')
        tfidf_model = TfidfModel(corpus, normalize=False)
        corpus = tfidf_model[corpus]
        logging.info('Transforming the corpus...')
        corpus = [tfidf_model[corpus] for corpus in corpus]
        model_name += 'tfidf_'

    if wiki_path is not None:
        model_name += 'wiki_'

    logging.info('Performing LDA on user corpus...')
    model, vectors = perform_lda(dictionary=dictionary, corpus=corpus, num_topics=num_topics, passes=passes,
                                 iterations=iterations, chunksize=chunksize, wiki_path=wiki_path)
    model.print_topics(topics=num_topics, topn=10)
    model.save('data/model/' + model_name + str(num_topics) + '.lda')
    enpickle(vectors, 'data/vector/' + model_name + str(num_topics) + '.pkl')


if __name__ == '__main__':
    # parameters
    num_topics = 100
    passes = 10
    iterations = 50
    chunksize = 2000
    # wiki_path = 'data/processed/wikis.pkl'
    wiki_path = None

    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    do_lda(num_topics=num_topics, passes=passes, iterations=iterations, chunksize=chunksize, tfidf=None, wiki_path=wiki_path)