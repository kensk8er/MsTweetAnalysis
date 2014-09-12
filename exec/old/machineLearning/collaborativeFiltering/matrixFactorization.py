"""
Executable file to perform matrix-factorization model.

* Not implemented yet
"""
import graphlab as gl
from sklearn.metrics import precision_score
from module.ML.CF import MatrixFactorization
from util.input import unpickle
from util.log import configure_log
from util.output import enpickle

__author__ = 'kensk8er'

if __name__ == '__main__':
    # variables
    reveal_ratio = 0.5
    min_app_num = 2

    # logging
    logging = configure_log(__file__)

    logging.info("Loading preference data and split it into train and test set...")
    pref_data = unpickle('data/converted/production/pref_data_sparse.pkl')
    # pref_sf = gl.load_sframe('data/converted/production/pref_data.sf')
    pref_sf = MatrixFactorization.array2random_explicit_sf(array=pref_data.tolil(), row_key='user_id', col_key='job_id',
                                                           val_key='rating')
    pref_sf.save('sandbox/pref.sf')

    (train_set, test_set) = MatrixFactorization.train_test_split(pref_sf=pref_sf, test_ratio=0.1)

    logging.info("Deleting users who have applied less than %s jobs." % min_app_num)
    test_set, test_user_ids = MatrixFactorization.remove_user_by_count(data_set=test_set, min_count=min_app_num)

    logging.info("Converting test set into array format...")
    test_set = MatrixFactorization.sf2array(sf=test_set, shape=pref_data.shape, row_key='user_id', col_key='job_id',
                                            val_key='rating')

    logging.info("Hide %s of ratings from test set..." % (1 - reveal_ratio))
    test_data, test_labels = MatrixFactorization.test_test_split(test_set=test_set, reveal_ratio=reveal_ratio,
                                                                 user_ids=test_user_ids)

    logging.info("Actual reveal ratio: %s" % (float(test_data.sum()) / (test_data.sum() + test_labels.sum())))

    logging.info("Converting test data into SFrame format...")
    test_data = MatrixFactorization.array2sf(array=test_data, row_key='user_id', col_key='job_id', val_key='rating')

    logging.info("Learning item-based collaborative filtering model...")
    item_sim_model = gl.recommender.create(observation_data=train_set.append(test_data), user_column='user_id',
                                           item_column='job_id', target_column='rating', method='item_similarity',
                                           similarity_type='jaccard', verbose=False)

    logging.info("Computing scores for test data...")
    job_num = len(pref_sf['job_id'].unique())
    scores = item_sim_model.recommend(users=test_user_ids, k=job_num, exclude_known=False)
    scores = scores[scores['score'] > 0]

    logging.info("Converting scores into array format...")
    scores = MatrixFactorization.sf2array(sf=scores, shape=pref_data.shape, row_key='user_id', col_key='job_id',
                                          val_key='score', dtype=float)

    logging.info("Deleting irrelevant rows from test labels")
    valid_user_ids = test_labels.sum(axis=1).nonzero()[0]
    scores = scores[valid_user_ids, :]
    test_labels = test_labels[valid_user_ids, :]

    logging.info("Computing possible labels...")
    possible_labels = set(pref_sf['job_id'].unique())

    logging.info("Predicting test data...")
    predictions = MatrixFactorization.predict(scores=scores, labels=test_labels, possible_labels=possible_labels)
    enpickle(predictions, 'sandbox/predictions.pkl')

    precision = precision_score(y_true=test_labels, y_pred=predictions, average='samples')
    logging.info("Precision: %s" % precision)
