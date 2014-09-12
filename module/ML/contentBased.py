"""
Self-defined functions for content-based filtering algorithms.

Algorithms: SVM, k-nearest neighbors, naive bayes
"""
from random import sample
import warnings
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from util.log import configure_log
from util.output import enpickle

__author__ = 'kensk8er'


class Base(object):
    @staticmethod
    def get_valid_job_ids(pref_data, min_app_num=1):
        valid_job_ids = []
        for job_id, app_num in enumerate(pref_data.sum(axis=0).T):  # take the transpose in order to iterate
            app_num = app_num[0, 0]
            if app_num >= min_app_num:
                valid_job_ids.append(job_id)
        return np.array(valid_job_ids)

    @staticmethod
    def get_valid_user_ids(pref_data, corpus_user_ids, min_app_num=1):
        valid_user_ids = []
        for user_id, app_num in enumerate(pref_data.sum(axis=1)):
            app_num = app_num[0, 0]
            if app_num >= min_app_num and corpus_user_ids.has_key(user_id):  # focus on users who have CV data
                valid_user_ids.append(user_id)
        return np.array(valid_user_ids)

    @staticmethod
    def get_labels(pref_data):  # TODO: this is slow!
        labels = []
        for row in xrange(pref_data.shape[0]):
            labels.append(pref_data.getrow(row).nonzero()[1].tolist())
        labels = tuple(labels)
        return labels

    @staticmethod
    def gen_user_profiles(valid_user_ids, user_vectors, corpus_id2user_id, user_data):
        user_profiles = []
        for user_id in valid_user_ids:
            user_datum = user_data[user_id]
            # integrate gender information
            user_profile = []
            user_profile.append(1) if user_datum['gender'] == 'male' else user_profile.append(0)
            user_profile.append(1) if user_datum['gender'] == 'female' else user_profile.append(0)

            corpus_id = corpus_id2user_id.index(user_id)
            user_profile.extend(user_vectors[corpus_id, :])
            user_profiles.append(user_profile)
        user_profiles = np.array(user_profiles)
        return user_profiles

    @staticmethod
    def gen_job_profiles(valid_job_ids, job_vectors, job_data):
        # trim the irrelevant or unwieldy columns from the job data
        new_job_data = []
        for job_datum in job_data:
            # TODO: re-consider these categorical features
            new_job_datum = {}
            new_job_datum['compensation_type'] = job_datum['compensation_type']
            new_job_datum['employment_type'] = job_datum['employment_type']
            new_job_datum['featured'] = job_datum['featured']
            new_job_datum['function_name'] = job_datum['function_name']
            new_job_datum['industry_name'] = job_datum['industry_name']
            new_job_data.append(new_job_datum)

        vectorizer = DictVectorizer(sparse=False)
        job_data = vectorizer.fit_transform(new_job_data)

        job_profiles = []
        for job_id in valid_job_ids:
            job_profile = job_data[job_id]
            corpus_id = job_id  # Note that every job has corpus, thus `job_id = corpus_id`
            job_profile = np.append(job_profile, job_vectors[corpus_id, :])
            job_profiles.append(job_profile)
        job_profiles = np.array(job_profiles)
        return job_profiles

    @staticmethod
    def compactify(pref_data, valid_user_ids, valid_job_ids):
        valid_row_indices = np.array(pref_data.sum(axis=1).nonzero()[0])[0]
        valid_col_indices = np.array(pref_data.sum(axis=0).nonzero()[1])[0]
        pref_data = csr_matrix(pref_data)[:, valid_col_indices]
        pref_data = coo_matrix(pref_data[valid_row_indices, :])
        valid_user_ids = valid_user_ids[valid_row_indices]
        valid_job_ids = valid_job_ids[valid_col_indices]
        return pref_data, valid_user_ids, valid_job_ids

    @staticmethod
    def label_propagation(labels, label_profiles, threshold):  # TODO: doesn't work well
        if threshold == 1:
            return labels  # Do nothing if the threshold = 1

        before_sum = labels.sum()
        similarities = cosine_similarity(label_profiles)
        propagation = {}
        for label_id, similarity in enumerate(similarities):
            propagation[label_id] = np.where(similarity > threshold)[0]

        new_labels = np.zeros(labels.shape, dtype=np.int)
        for user_id, label in enumerate(labels):
            for label_id in label.nonzero()[0]:
                new_labels[user_id, propagation[label_id]] = 1
        after_sum = new_labels.sum()
        return new_labels, before_sum, after_sum


class SVM(Base):
    @staticmethod
    def predict(classifier, data, labels, possible_labels, binarizer):
        confidences = classifier.decision_function(data)
        confidences[np.where(confidences == 0)[0], np.where(confidences == 0)[1]] = -100  # don't predict unseen labels
        predictions = []
        for confidence, label in zip(confidences, labels):
            label_num = sum(label)
            sample_num = label_num if len(possible_labels) >= 2 * label_num else len(possible_labels) - label_num
            candidates = np.append(label.nonzero()[0],
                                   sample(possible_labels.difference(label.nonzero()[0]), sample_num))
            predictions.append(tuple(candidates[confidence[candidates].argsort()[::-1]][
                                     :label_num]))  # assume the number of correct labels is given
        return binarizer.transform(predictions)

    @staticmethod
    def predict_k(classifier, data, binarizer, k=1):
        confidences = classifier.decision_function(data)
        confidences[np.where(confidences == 0)[0], np.where(confidences == 0)[1]] = -100  # don't predict unseen labels
        predictions = []
        for confidence in confidences:
            predictions.append(tuple(confidence.argsort()[::-1][:k]))
        return binarizer.transform(predictions)

    @classmethod
    def run_svm(cls, train, test, C, binarizer, labels, intercept):
        """


        :param binarizer:
        :param labels:
        :param train:
        :param test:
        :param C:
        """
        # logging
        logging = configure_log(__file__)

        logging.info("C = %s" % (str(C)))
        logging.info('Fitting Linear SVM...')
        train_data, train_labels = train
        test_data, test_labels = test

        dual = False if train_data.shape[0] > train_data.shape[1] else True
        classifier = OneVsRestClassifier(LinearSVC(dual=dual, class_weight=None, C=C, intercept_scaling=intercept))  # C -> inf = hard-margin
        with warnings.catch_warnings():  # FIXME: split the data set in a way that the train set has every label
            warnings.simplefilter("ignore")
            classifier.fit(train_data, train_labels)

        possible_labels = set()
        [map(possible_labels.add, row) for row in [label.nonzero()[0] for label in labels]]

        seen_labels = set()
        [map(seen_labels.add, row) for row in [label.nonzero()[0] for label in train_labels]]

        logging.info('Predicting test set...')
        test_predictions = cls.predict(classifier=classifier, data=test_data, labels=test_labels,
                                       possible_labels=possible_labels, binarizer=binarizer)
        # test_predictions = cls.predict_k(classifier=classifier, data=test_data, k=100, binarizer=binarizer)

        # logging.info('Predicting train set...')
        # train_predictions = cls.predict(classifier=classifier, data=train_data, labels=train_labels,
        #                                 possible_labels=possible_labels, binarizer=binarizer)

        precision = precision_score(y_true=test_labels, y_pred=test_predictions, average='samples')
        # (precision, recall, f, support) = precision_recall_fscore_support(y_true=test_labels, y_pred=test_predictions,
        #                                                                   average='samples')
        # train_precision = precision_score(y_true=train_labels, y_pred=train_predictions, average='samples')

        # return test_precision
        return precision#, recall, f, support


class KNN(Base):
    @staticmethod
    def neighbor_fit(similarities, labels, k):
        confidences = []
        for similarity in similarities.T:
            indices = similarity.argsort()[::-1][:k]
            confidences.append(np.array(labels[indices, :].sum(axis=0))[0])
        return np.array(confidences)

    @staticmethod
    def neighbor_weight_fit(similarities, labels, k):
        confidences = []
        for similarity in similarities.T:
            indices = similarity.argsort()[::-1][:k]
            weights = [1. / (rank + 1) for rank in xrange(k)]
            confidences.append(np.array(labels[indices, :].multiply(np.reshape(weights, (k, 1))).sum(axis=0))[
                0])  # TODO: this is significantly slow, rewrite in Cython
        return np.array(confidences)

    @staticmethod
    def radius_fit(similarities, labels, threshold):
        confidences = []
        for similarity in similarities.T:
            indices = similarity.argsort()[::-1]
            indices_of_indices = np.where(similarity[indices] > threshold)[0]
            if len(indices_of_indices) == 0:
                confidences.append(np.zeros((labels.shape[1],), dtype=int))
            else:
                indices = indices[indices_of_indices]
                confidences.append(np.array(labels[indices, :].sum(axis=0))[0])
        return np.array(confidences)

    @staticmethod
    def radius_weight_fit(similarities, labels, threshold, power=2):
        confidences = []
        for similarity in similarities.T:
            indices = similarity.argsort()[::-1]
            indices_of_indices = np.where(similarity[indices] > threshold)[0]
            if len(indices_of_indices) == 0:
                confidences.append(np.zeros((labels.shape[1],), dtype=int))
            else:
                indices = indices[indices_of_indices]
                confidences.append(
                    np.array(labels[indices, :].multiply(np.reshape(similarity[indices] ** power, (len(indices), 1))).sum(
                        axis=0))[0])  # TODO: this is significantly slow, rewrite in Cython
        return np.array(confidences)

    @staticmethod
    def predict(confidences, labels, possible_labels):
        predictions = np.zeros(labels.shape, dtype=int)
        for row_num, (confidence, label) in enumerate(zip(confidences, labels)):
            # this noise is for preventing simply choosing the last two candidates in the candidates numpy-array.
            randomized_confidence = np.array(confidence, dtype=float) + np.random.rand(len(confidence)) * 0.000001
            label_num = label.sum()
            sample_num = label_num if len(possible_labels) >= 2 * label_num else len(possible_labels) - label_num
            candidates = np.append(label.nonzero()[1], sample(possible_labels.difference(label.nonzero()[1]), sample_num))
            predictions[row_num, candidates[randomized_confidence[candidates].argsort()[::-1]][:label_num]] = 1
        return predictions


class Popularity(Base):
    @staticmethod
    def get_popularity(labels):
        return labels.sum(axis=0)

    @staticmethod
    def predict(popularity, labels, possible_labels, binarizer):
        predictions = []
        # this noise is for preventing simply choosing the last two candidates in the candidates numpy-array.
        popularity = np.array(popularity, dtype=float) + np.random.rand(len(popularity)) * 0.000001
        for label in labels:
            label_num = sum(label)
            sample_num = label_num if len(possible_labels) >= 2 * label_num else len(possible_labels) - label_num
            candidates = np.append(label.nonzero()[0], sample(possible_labels.difference(label.nonzero()[0]), sample_num))
            predictions.append(tuple(candidates[popularity[candidates].argsort()[::-1]][:label_num]))
        return binarizer.transform(predictions)

    @staticmethod
    def predict_k(popularity, sample_num, binarizer, k=1):
        predictions = [tuple(popularity.argsort()[::-1][:k]) for i in xrange(sample_num)]
        return binarizer.transform(predictions)


class NaiveBayes(Base):
    @staticmethod
    def predict(classifier, data, labels, possible_labels, binarizer):
        with warnings.catch_warnings():  # FIXME: split the data set in a way that the train set has every label
            warnings.simplefilter("ignore")
            confidences = classifier.predict_proba(data)
        predictions = []
        for confidence, label in zip(confidences, labels):
            label_num = sum(label)
            sample_num = label_num if len(possible_labels) >= 2 * label_num else len(possible_labels) - label_num
            candidates = np.append(label.nonzero()[0],
                                   sample(possible_labels.difference(label.nonzero()[0]), sample_num))
            predictions.append(tuple(candidates[confidence[candidates].argsort()[::-1]][
                                     :label_num]))  # assume the number of correct labels is given
        return binarizer.transform(predictions)

    @classmethod
    def run_naive_bayes(cls, train, test, binarizer, labels, alpha):
        # logging
        logging = configure_log(__file__)

        logging.info("alpha = %s" % (str(alpha)))
        logging.info('Fitting Naive Bayes...')
        train_data, train_labels = train
        test_data, test_labels = test

        classifier = OneVsRestClassifier(MultinomialNB(alpha=alpha, fit_prior=True, class_prior=None))
        with warnings.catch_warnings():  # FIXME: split the data set in a way that the train set has every label
            warnings.simplefilter("ignore")
            classifier.fit(train_data, train_labels)

        possible_labels = set()
        [map(possible_labels.add, row) for row in [label.nonzero()[0] for label in labels]]

        logging.info('Predicting test set...')
        test_predictions = cls.predict(classifier=classifier, data=test_data, labels=test_labels,
                                        possible_labels=possible_labels, binarizer=binarizer)

        # logging.info('Predicting train set...')
        # train_predictions = cls.predict(classifier=classifier, data=train_data, labels=train_labels,
        #                                  possible_labels=possible_labels, binarizer=binarizer)

        test_precision = precision_score(y_true=test_labels, y_pred=test_predictions, average='samples')
        # train_precision = precision_score(y_true=train_labels, y_pred=train_predictions, average='samples')

        # return train_precision, test_precision
        return test_precision
