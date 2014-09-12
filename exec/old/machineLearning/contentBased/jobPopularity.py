"""
Baseline algorithm using popularity based recommendation.
"""
from random import sample
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from module.ML.contentBased import Popularity
from util.input import unpickle
from util.log import configure_log
from util.output import enpickle

__author__ = 'kensk8er'

if __name__ == '__main__':
    # logging
    logging = configure_log(__file__)
    min_user_app_num = 2

    logging.info('Loading preference data...')
    pref_data = unpickle('data/converted/production/pref_data_sparse.pkl')

    corpus_id2user_id = unpickle('data/dictionary/corpus_id2user_id.pkl')
    try:
        valid_user_ids = unpickle('tmp/jobPopularity_valid_user_ids.pkl')
        logging.info('The number of users: %s' % len(valid_user_ids))
    except:
        logging.info('Detecting users who haven\'t applied more than or equal to one job listing...')
        corpus_user_ids = {corpus_id2user_id[i]: 0 for i in
                           xrange(len(corpus_id2user_id))}  # trick to make the loop faster
        valid_user_ids = Popularity.get_valid_user_ids(pref_data=pref_data, corpus_user_ids=corpus_user_ids,
                                                       min_app_num=min_user_app_num)
        logging.info('The number of users: %s' % len(valid_user_ids))
        enpickle(valid_user_ids, 'tmp/jobPopularity_valid_user_ids.pkl')

    logging.info('Deleting users which haven\'t applied or doesn\'t have CV data...')
    pref_data = coo_matrix(csr_matrix(pref_data)[valid_user_ids, :])

    logging.info('Deleting jobs which haven\'t been applied at all...')
    valid_job_ids = Popularity.get_valid_job_ids(pref_data, min_app_num=1)
    logging.info('The number of jobs: %s' % len(valid_job_ids))
    pref_data = coo_matrix(csr_matrix(pref_data)[:, valid_job_ids])

    logging.info('Deleting all zero rows and columns from preference data...')
    pref_data, valid_user_ids, valid_job_ids = Popularity.compactify(pref_data=pref_data, valid_user_ids=valid_user_ids,
                                                                     valid_job_ids=valid_job_ids)
    logging.info('The number of users: %s' % len(valid_user_ids))
    logging.info('The number of jobs: %s' % len(valid_job_ids))

    try:
        labels = unpickle('tmp/jobPopularity_labels.pkl')
    except:
        logging.info('Getting labels...')
        labels = Popularity.get_labels(pref_data=pref_data.T)
        enpickle(labels, 'tmp/jobPopularity_labels.pkl')

    logging.info('Splitting data into train and test set...')
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    train_labels, test_labels = train_test_split(labels, test_size=0.1, random_state=42)

    possible_labels = set()
    [map(possible_labels.add, row) for row in [label.nonzero()[0] for label in labels]]

    popularity = Popularity.get_popularity(labels=train_labels)

    avg_test_precision = 0.
    count = 0
    for i in range(10):
        test_predictions = Popularity.predict(popularity=popularity, labels=test_labels,
                                              possible_labels=possible_labels, binarizer=mlb)
        test_precision = precision_score(y_true=test_labels, y_pred=test_predictions, average='samples')
        logging.info("test precision: %s" % (test_precision))
        avg_test_precision += test_precision
        count += 1

    avg_test_precision /= count
    logging.info('10 average precision: %s' % avg_test_precision)
