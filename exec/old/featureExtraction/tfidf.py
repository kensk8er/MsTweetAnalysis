"""
Executable file for TF-IDF (Term frequency - Inverse Document Frequency).

When running this file, 'working directory' need to be specified as Project Root (EnternshipsJobRecommendation).
"""
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from util.corpus import corpus2vector, filter_term
from util.log import configure_log
from util.output import enpickle

__author__ = 'kensk8er'


if __name__ == '__main__':
    # variables
    no_below = 5
    no_above = 0.2
    keep_n = 2000

    # logging
    logging = configure_log(__file__)

    logging.info('Loading user dictionary...')
    user_dict = Dictionary.load('data/dictionary/user_(NN).dict')
    user_corpus = user_dict.corpus

    logging.info('Reduce the number of terms...')
    user_dict.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    user_corpus = filter_term(corpora=user_corpus, valid_term_ids=user_dict.token2id.values())

    logging.info('Computing TF-IDF...')
    tfidf_user = TfidfModel(user_corpus, normalize=False)
    logging.info('Transforming the user corpus...')
    user_corpus = [tfidf_user[corpus] for corpus in user_corpus]
    tfidf_user.save('data/feature/tfidf/model/user.tfidf')

    logging.info('Converting corpus into vector format...')
    user_vectors = corpus2vector(corpora=user_corpus, num_terms=len(user_dict.token2id))
    enpickle(user_vectors, 'data/feature/tfidf/user.pkl')

    logging.info('Loading job dictionary...')
    job_dict = Dictionary.load('data/dictionary/job_(NN).dict')
    job_corpus = job_dict.corpus

    logging.info('Computing TF-IDF...')
    tfidf_job = TfidfModel(job_corpus, normalize=False)
    logging.info('Transforming the job corpus...')
    job_corpus = [tfidf_job[corpus] for corpus in job_corpus]
    tfidf_job.save('data/feature/tfidf/model/job.tfidf')

    logging.info('Converting corpus into vector format...')
    job_vectors = corpus2vector(corpora=job_corpus, num_terms=len(job_dict.id2token))
    enpickle(job_vectors, 'data/feature/tfidf/job.pkl')