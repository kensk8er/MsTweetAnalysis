"""
Executable file to perform user-based SVM model.

user-based SVM uses extracted text features as part of user-profile. Each user is viewed as a sample, and each job is
viewed as a label.
"""
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize
from module.ML.contentBased import SVM
from util.input import unpickle
from util.log import configure_log
from util.output import enpickle

__author__ = 'kensk8er'

if __name__ == '__main__':
    # parameters
    user_feature_path = 'data/feature/plsa/user_200.pkl'
    min_job_app_num = 2
    min_user_app_num = 1
    label_propagation = False
    job_feature_path = 'data/feature/lda/job_50.pkl'
    threshold = 0.9

    # logging
    logging = configure_log(__file__)
    logging.info("variables: user_feature_path = %s, min_job_app_num = %s" % (user_feature_path, min_job_app_num))

    logging.info('Loading preference data...')
    pref_data = unpickle('data/converted/production/pref_data_sparse.pkl')

    corpus_id2user_id = unpickle('data/dictionary/corpus_id2user_id.pkl')
    try:
        valid_user_ids = unpickle('tmp/userSVM_valid_user_ids.pkl')
        logging.info('The number of users: %s' % len(valid_user_ids))
    except:
        logging.info("Detecting users who haven't applied more than or equal to one job listing...")
        corpus_user_ids = {corpus_id2user_id[i]: 0 for i in
                           xrange(len(corpus_id2user_id))}  # trick to make the loop faster
        valid_user_ids = SVM.get_valid_user_ids(pref_data=pref_data, corpus_user_ids=corpus_user_ids,
                                                min_app_num=min_user_app_num)
        logging.info('The number of users: %s' % len(valid_user_ids))
        enpickle(valid_user_ids, 'tmp/userSVM_valid_user_ids.pkl')

    logging.info("Deleting users which haven't applied or doesn't have CV data...")
    pref_data = coo_matrix(csr_matrix(pref_data)[valid_user_ids, :])

    logging.info("Deleting jobs which haven't been applied at all...")
    valid_job_ids = SVM.get_valid_job_ids(pref_data, min_app_num=min_job_app_num)
    logging.info('The number of jobs: %s' % len(valid_job_ids))
    pref_data = coo_matrix(csr_matrix(pref_data)[:, valid_job_ids])

    logging.info('Deleting all zero rows and columns from preference data...')
    pref_data, valid_user_ids, valid_job_ids = SVM.compactify(pref_data=pref_data, valid_user_ids=valid_user_ids,
                                                              valid_job_ids=valid_job_ids)
    logging.info('The number of users: %s' % len(valid_user_ids))
    logging.info('The number of jobs: %s' % len(valid_job_ids))

    try:
        labels = unpickle('tmp/userSVM_labels.pkl')
    except:
        logging.info('Getting labels...')
        labels = SVM.get_labels(pref_data=pref_data)
        enpickle(labels, 'tmp/userSVM_labels.pkl')

    logging.info('Loading user data...')
    user_data = unpickle('data/converted/production/user_data.pkl')

    logging.info('Generating user profile...')
    user_vectors = unpickle(user_feature_path)
    user_profiles = SVM.gen_user_profiles(valid_user_ids=valid_user_ids, user_vectors=user_vectors,
                                          corpus_id2user_id=corpus_id2user_id, user_data=user_data)

    logging.info('Splitting data into train and test set...')
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    train_user_profiles, test_user_profiles, train_labels, test_labels = train_test_split(
        user_profiles, labels, test_size=0.1, random_state=42)

    if label_propagation is True:
        logging.info('Loading job data...')
        job_data = unpickle('data/converted/production/job_data.pkl')

        logging.info('Generating job profile...')
        job_vectors = unpickle(job_feature_path)

        logging.info('Normalizing job text data...')
        job_vectors = normalize(job_vectors, norm='l2')
        job_profiles = SVM.gen_job_profiles(valid_job_ids=valid_job_ids, job_vectors=job_vectors, job_data=job_data)

        logging.info('Propagating train labels...')
        train_labels, before_sum, after_sum = SVM.label_propagation(labels=train_labels, label_profiles=job_profiles,
                                                                    threshold=threshold)
        logging.info("Before propagation: %s, After propagation: %s" % (before_sum, after_sum))

    logging.info('Standardizing user profiles...')
    user_scaler = StandardScaler()
    user_scaler.fit(train_user_profiles)
    train_user_profiles, test_user_profiles = map(user_scaler.transform, (train_user_profiles, test_user_profiles))

    avg_test_precision = 0.
    count = 0
    # for i in range(10):
    for i in range(10):
        # for C in [10 ** (-i / 10.) for i in range(50, 70)]:  # TODO: use grid search and cross-validation here
        # for intercept in [i / 10. for i in range(10, 30)]:
        # train_precision, test_precision = run_svm(train=(train_user_profiles, train_labels),
        # test=(test_user_profiles, test_labels), C=C, mlb=mlb, labels=labels)
        precision = SVM.run_svm(train=(train_user_profiles, train_labels),
                                test=(test_user_profiles, test_labels), C=10 ** -6, binarizer=mlb,
                                labels=labels, intercept=1.7)
        # precision, recall, f, support = SVM.run_svm(train=(train_user_profiles, train_labels),
        #                                             test=(test_user_profiles, test_labels), C=10 ** -6, binarizer=mlb,
        #                                             labels=labels, intercept=1.7)

        # logging.info("train precision: %s" % (train_precision))
        logging.info("test precision: %s" % (precision))
        # logging.info("test precision: %s" % (precision))
        # logging.info("test recall: %s" % (recall))
        # logging.info("test f: %s" % (f))
        # logging.info("test support: %s" % (support))

        # avg_test_precision += test_precision
        avg_test_precision += precision
        count += 1

    avg_test_precision /= count
    logging.info('10 average precision: %s' % avg_test_precision)