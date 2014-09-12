"""
Executable file to perform job-based SVM model.

job-based SVM uses extracted text features as part of job-profile. Each job is viewed as a sample, and each user is
viewed as a label.
"""
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from module.ML.contentBased import SVM
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
        valid_user_ids = unpickle('tmp/jobSVM_valid_user_ids.pkl')
        logging.info('The number of users: %s' % len(valid_user_ids))
    except:
        logging.info('Detecting users who haven\'t applied more than or equal to one job listing...')
        corpus_user_ids = {corpus_id2user_id[i]: 0 for i in xrange(len(corpus_id2user_id))}  # trick to make the loop faster
        valid_user_ids = SVM.get_valid_user_ids(pref_data, corpus_user_ids=corpus_user_ids, min_app_num=min_user_app_num)
        logging.info('The number of users: %s' % len(valid_user_ids))
        enpickle(valid_user_ids, 'tmp/jobSVM_valid_user_ids.pkl')

    logging.info('Deleting users which haven\'t applied or doesn\'t have CV data...')
    pref_data = coo_matrix(csr_matrix(pref_data)[valid_user_ids, :])

    logging.info('Detecting jobs which haven\'t been applied by more than or equal to one user...')
    valid_job_ids = SVM.get_valid_job_ids(pref_data=pref_data, min_app_num=1)
    logging.info('The number of jobs: %s' % len(valid_job_ids))
    pref_data = coo_matrix(csr_matrix(pref_data)[:, valid_job_ids])

    logging.info('Deleting all zero rows and columns from preference data...')
    pref_data, valid_user_ids, valid_job_ids = SVM.compactify(pref_data=pref_data, valid_user_ids=valid_user_ids,
                                                              valid_job_ids=valid_job_ids)
    logging.info('The number of users: %s' % len(valid_user_ids))
    logging.info('The number of jobs: %s' % len(valid_job_ids))

    try:
        labels = unpickle('tmp/jobSVM_labels.pkl')
    except:
        logging.info('Getting labels...')
        labels = SVM.get_labels(pref_data=pref_data.T)
        enpickle(labels, 'tmp/jobSVM_labels.pkl')

    logging.info('Loading job data...')
    job_data = unpickle('data/converted/production/job_data.pkl')

    logging.info('Generating job profile...')
    job_vectors = unpickle(job_feature_path)
    job_profiles = SVM.gen_job_profiles(valid_job_ids=valid_job_ids, job_vectors=job_vectors, job_data=job_data)

    logging.info('Splitting data into train and test set...')
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    train_job_profiles, test_job_profiles, train_labels, test_labels = train_test_split(
        job_profiles, labels, test_size=0.1, random_state=42)

    logging.info('Standardizing job profiles...')
    job_scaler = StandardScaler()
    job_scaler.fit(train_job_profiles)
    train_job_profiles, test_job_profiles = map(job_scaler.transform, (train_job_profiles, test_job_profiles))

    avg_test_precision = 0.
    count = 0
    # for C in [10 ** -i for i in range(6, 10)]:  # TODO: use grid search and cross-validation here
    for i in range(10):
        test_precision = SVM.run_svm(train=(train_job_profiles, train_labels),
                                     test=(test_job_profiles, test_labels), C=10**-6, binarizer=mlb,
                                     labels=labels, intercept=1.7)

        # logging.info("train precision: %s" % (train_precision))
        logging.info("test precision: %s" % (test_precision))
        # logging.info("train job_num: %s" % (len(train_labels)))
        # logging.info("test job_num: %s\n" % (len(test_labels)))

        avg_test_precision += test_precision
        count += 1

    avg_test_precision /= count
    logging.info('10 average precision: %s' % avg_test_precision)
