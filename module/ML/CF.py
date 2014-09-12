"""
Self-defined functions for collaborative-filtering algorithms.

Algorithms: item-based, user-based, etc.
"""
from random import sample
from graphlab import SArray, SFrame
import numpy as np
import graphlab.aggregate as agg

__author__ = 'kensk8er'


class Base(object):
    @staticmethod
    def train_test_split(pref_sf, test_ratio):
        user_ids = list(pref_sf['user_id'].unique())
        test_user_ids = sample(user_ids, int(len(user_ids) * test_ratio))
        train_user_ids = list(set(user_ids).difference(test_user_ids))
        train_set = pref_sf.filter_by(train_user_ids, 'user_id')
        test_set = pref_sf.filter_by(test_user_ids, 'user_id')
        return train_set, test_set

    @staticmethod
    def remove_user_by_count(data_set, min_count=2):
        user_count = data_set.groupby(key_columns='user_id', operations={'count': agg.COUNT()})
        user_ids = list(user_count[user_count['count'] >= min_count]['user_id'])
        return data_set.filter_by(user_ids, 'user_id'), user_ids

    @staticmethod
    def test_test_split(test_set, reveal_ratio, user_ids):
        test_data = np.zeros(test_set.shape, dtype=int)
        test_labels = np.zeros(test_set.shape, dtype=int)
        for user_id in user_ids:
            label_num = test_set[user_id, :].sum()
            sample_num = int(round(label_num * reveal_ratio))
            if sample_num == label_num:
                sample_num -= 1
            candidates = set(test_set[user_id, :].nonzero()[0])
            sampled = sample(candidates, sample_num)
            test_data[user_id, sampled] = 1
            test_labels[user_id, list(candidates.difference(sampled))] = 1
        return test_data, test_labels

    @staticmethod
    def sf2array(sf, shape, row_key, col_key, val_key=None, dtype=int):
        sf = list(sf)
        array = np.zeros(shape, dtype=dtype)
        for record in sf:
            array[record[row_key], record[col_key]] = 1 if val_key is None else record[val_key]
        return array

    @staticmethod
    def array2sf(array, row_key, col_key, val_key=None):
        (rows, cols) = map(SArray, array.nonzero())
        if val_key is None:
            return SFrame({row_key: rows, col_key: cols})
        else:
            vals = SArray(array[array.nonzero()])
            return SFrame({row_key: rows, col_key: cols, val_key: vals})

    @staticmethod
    def array2explicit_sf(array, row_key, col_key, val_key):  # TODO: super slow, can't be used practically
        rows = np.zeros((array.shape[0] * array.shape[1]), dtype=int)
        for row in xrange(array.shape[0]):
            rows[row * array.shape[1]: (row + 1) * array.shape[1]] = row
        cols = np.tile([i for i in xrange(array.shape[1])], array.shape[0])
        vals = np.zeros((array.shape[0] * array.shape[1]), dtype=int)
        for row_index, col_index in zip(array.nonzero()[0], array.nonzero()[1]):
            vals[row_index * array.shape[1] + col_index] = 1
        return SFrame({row_key: SArray(rows), col_key: SArray(cols), val_key: SArray(vals)})

    @staticmethod
    def array2random_explicit_sf(array, row_key, col_key, val_key, density=0.001):  # TODO: still slow, rewrite in cython
        element_num = array.shape[0] * array.shape[1]
        non_zeros = []
        for row_index, col_index in zip(array.nonzero()[0], array.nonzero()[1]):
            non_zeros.append(row_index * array.shape[1] + col_index)
        sample_num = int((element_num - len(non_zeros)) * density)
        selected = sample(xrange(element_num), sample_num)
        selected_index = {i: 0 for i in selected}
        for non_zero in non_zeros:
            if selected_index.has_key(non_zero):
                selected.remove(non_zero)
        selected_row = [i // array.shape[1] for i in selected]
        selected_col = [i % array.shape[1] for i in selected]
        (rows, cols) = array.nonzero()
        vals = [1 for i in xrange(len(non_zeros))]
        rows = np.append(rows, selected_row)
        cols = np.append(cols, selected_col)
        vals = np.append(vals, [0 for i in xrange(len(selected))])

        return SFrame({row_key: SArray(rows), col_key: SArray(cols), val_key: SArray(vals)})

    @staticmethod
    def predict(scores, labels, possible_labels):
        predictions = np.zeros(labels.shape, dtype=int)
        for row, (score, label) in enumerate(zip(scores, labels)):
            print "\r%s / %s" % (row + 1, len(scores)),
            label_num = label.sum()
            if label_num > 0:
                sample_num = label_num if len(possible_labels) >= 2 * label_num else len(possible_labels) - label_num
                score = np.array(score, dtype=float) + np.random.rand(len(score)) * 0.000001
                candidates = np.append(label.nonzero()[0],
                                       sample(possible_labels.difference(label.nonzero()[0]), sample_num))
                predictions[row, candidates[score[candidates].argsort()[::-1]][:label_num]] = 1
        print '\nDone.'
        return predictions


class ItemBased(Base):
    pass


class ItemPopularity(Base):
    pass


class MatrixFactorization(Base):
    pass


class Hybrid(Base):
    pass