"""
k-nearest neighbor classifier
"""
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from module.ML.contentBased import KNN
from util.input import unpickle
from util.log import configure_log
from util.output import enpickle

__author__ = 'kensk8er'

if __name__ == '__main__':
    # parameters
    job_feature_path = 'data/feature/lda/job_80.pkl'
    min_user_app_num = 2

    # logging
    logging = configure_log(__file__)
    logging.info("variables: job_feature_path = %s, min_user_app_num = %s" % (job_feature_path, min_user_app_num))

    logging.info('Loading preference data...')
    pref_data = unpickle('data/converted/production/pref_data_sparse.pkl')

    corpus_id2user_id = unpickle('data/dictionary/corpus_id2user_id.pkl')
    try:
        valid_user_ids = unpickle('tmp/jobKNN_valid_user_ids.pkl')
        logging.info('The number of users: %s' % len(valid_user_ids))
    except:
        logging.info("Detecting users who haven't applied more than or equal to one job listing...")
        corpus_user_ids = {corpus_id2user_id[i]: 0 for i in
                           xrange(len(corpus_id2user_id))}  # trick to make the loop faster
        valid_user_ids = KNN.get_valid_user_ids(pref_data=pref_data, corpus_user_ids=corpus_user_ids,
                                                min_app_num=min_user_app_num)
        logging.info('The number of users: %s' % len(valid_user_ids))
        enpickle(valid_user_ids, 'tmp/jobKNN_valid_user_ids.pkl')

    logging.info("Deleting users which haven't applied or doesn't have CV data...")
    pref_data = coo_matrix(csr_matrix(pref_data)[valid_user_ids, :])

    logging.info("Deleting jobs which haven't been applied at all...")
    valid_job_ids = KNN.get_valid_job_ids(pref_data, min_app_num=1)
    logging.info('The number of jobs: %s' % len(valid_job_ids))
    pref_data = coo_matrix(csr_matrix(pref_data)[:, valid_job_ids])

    logging.info('Deleting all zero rows and columns from preference data...')
    pref_data, valid_user_ids, valid_job_ids = KNN.compactify(pref_data=pref_data, valid_user_ids=valid_user_ids,
                                                              valid_job_ids=valid_job_ids)
    logging.info('The number of users: %s' % len(valid_user_ids))
    logging.info('The number of jobs: %s' % len(valid_job_ids))

    logging.info('Loading job data...')
    job_data = unpickle('data/converted/production/job_data.pkl')

    logging.info('Generating job profile...')
    job_vectors = unpickle(job_feature_path)

    logging.info('Normalizing job text data...')
    job_vectors = normalize(job_vectors, norm='l2')

    job_profiles = KNN.gen_job_profiles(valid_job_ids=valid_job_ids, job_vectors=job_vectors, job_data=job_data)
    # job_profiles[:, :2] *= gender_weight  # adjust the weight for gender

    logging.info('Splitting data into train and test set...')
    train_job_profiles, test_job_profiles, train_labels, test_labels = train_test_split(
        job_profiles, pref_data.T, test_size=0.1, random_state=42)

    # logging.info('Reducing the dimensions of features...')
    # pca = PCA(n_components=30)
    # pca.fit(train_user_profiles)
    # train_user_profiles, test_user_profiles = map(pca.transform, (train_user_profiles, test_user_profiles))

    logging.info('Computing similarities between train data...')
    similarity = cosine_similarity(train_job_profiles, test_job_profiles)

    logging.info('Checking possible labels...')
    try:
        possible_labels = unpickle('tmp/jobKNN_possible_labels.pkl')
    except:
        possible_labels = set()
        [map(possible_labels.add, label) for label in [row.nonzero()[1] for row in pref_data.T.tocsr()]]  # TODO: this is slow
        enpickle(possible_labels, 'tmp/jobKNN_possible_labels.pkl')

    # k = 1500
    avg_precision = 0.
    count = 0
    # for k in range(150, 601, 50):
    for i in range(10):
        k = 300
        # threshold = i / 100.
        logging.info('Learning %s-NN...' % k)
        # logging.info('Learning %s-NN...' % threshold)
        # confidences = neighbor_weight_fit(similarities=similarity, labels=train_labels, threshold=threshold, power=3)
        confidences = KNN.neighbor_fit(similarities=similarity, labels=train_labels, k=k)

        logging.info('Predicting labels...')
        predicted = KNN.predict(confidences=confidences, labels=test_labels, possible_labels=possible_labels)
        precision = precision_score(y_true=np.array(test_labels.todense()), y_pred=predicted, average='samples')
        logging.info("precision: %s" % precision)
        avg_precision += precision
        count += 1

    avg_precision /= count
    logging.info('10 average precision: %s\n' % avg_precision)