"""
Executable file for PLSA (Probabilistic Latent Semantic Analysis).

When running this file, 'working directory' need to be specified as Project Root (EnternshipsJobRecommendation).
"""
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
from util.input import unpickle
from util.log import configure_log
from util.output import enpickle

__author__ = 'kensk8er'


def perform_plsa(dictionary, corpus, num_topics, wiki_path=None, passes=1, iterations=50, chunksize=200):
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

    plsa_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes,
                          iterations=iterations, alpha=1., eta=1., chunksize=chunksize)
    doc_vectors = plsa_model.inference(corpus)[0]
    doc_vectors = doc_vectors / doc_vectors.sum(axis=1).reshape(doc_vectors.shape[0], 1)

    return plsa_model, doc_vectors


def do_user(num_topics, passes, iterations, chunksize, tfidf):
    model_name = 'user_'

    logging.info('Loading user dictionary...')
    user_dict = corpora.Dictionary.load('data/dictionary/user_(NN).dict')
    user_corpus = user_dict.corpus

    if tfidf is True:
        logging.info('Computing TF-IDF...')
        tfidf_model = TfidfModel(user_corpus, normalize=False)
        user_corpus = tfidf_model[user_corpus]
        logging.info('Transforming the corpus...')
        user_corpus = [tfidf_model[corpus] for corpus in user_corpus]
        model_name += 'tfidf_'

    logging.info('Performing PLSA on user corpus...')
    user_plsa, user_vectors = perform_plsa(dictionary=user_dict, corpus=user_corpus, num_topics=num_topics,
                                           passes=passes, iterations=iterations, chunksize=chunksize)
    user_plsa.print_topics(topics=num_topics, topn=10)
    user_plsa.save('data/feature/plsa/model/' + model_name + str(num_topics) + '.lda')
    enpickle(user_vectors, 'data/feature/plsa/' + model_name + str(num_topics) + '.pkl')


def do_job(num_topics, passes, iterations, chunksize, tfidf):
    model_name = 'job_'

    logging.info('Loading job dictionary...')
    job_dict = corpora.Dictionary.load('data/dictionary/job_(NN).dict')
    job_corpus = job_dict.corpus

    logging.info('Performing PLSA on job corpus...')
    job_plsa, job_vectors = perform_plsa(dictionary=job_dict, corpus=job_corpus, num_topics=num_topics, passes=passes,
                                         iterations=iterations, chunksize=chunksize)
    job_plsa.print_topics(topics=num_topics, topn=10)
    job_plsa.save('data/feature/plsa/model/' + model_name + str(num_topics) + '.lda')
    enpickle(job_vectors, 'data/feature/plsa/' + model_name + str(num_topics) + '.pkl')


if __name__ == '__main__':
    # parameters
    user = True
    job = False
    user_num_topics = 5
    job_num_topics = 80
    passes = 2
    iterations = 50
    chunksize = 2000
    tfidf = False

    # logging
    logging = configure_log(__file__)

    logging.info(
        "variables: user = %s, job = %s, user_num_topics = %s, job_num_topics = %s, passes = %s, iterations = %s, chunksize = %s, tfidf = %s" % (
            user, job, user_num_topics, job_num_topics, passes, iterations, chunksize, tfidf))

    if user is True:
        do_user(num_topics=user_num_topics, passes=passes, iterations=iterations, chunksize=chunksize, tfidf=tfidf)
    if job is True:
        do_job(num_topics=job_num_topics, passes=passes, iterations=iterations, chunksize=chunksize, tfidf=tfidf)

