"""
Generating preference data in the SFrame format.
"""
from graphlab import SArray, SFrame
from module.ML.contentBased import Base
from util.input import unpickle
from util.log import configure_log

__author__ = 'kensk8er'


def gen_pref_sframe():
    pref_data = unpickle('data/converted/production/pref_data_sparse.pkl')
    (user_ids, job_ids) = map(SArray, pref_data.nonzero())
    pref_sf = SFrame({'user_id': user_ids, 'job_id': job_ids})
    pref_sf.save('data/converted/production/pref_data.sf')


def gen_user_sframe():
    pref_data = unpickle('data/converted/production/pref_data_sparse.pkl')
    user_feature_path = 'data/feature/lda/user_50.pkl'
    min_user_app_num = 1

    logging.info('Loading user data...')
    user_data = unpickle('data/converted/production/user_data.pkl')

    logging.info('Generating user profile...')
    user_vectors = unpickle(user_feature_path)

    corpus_id2user_id = unpickle('data/dictionary/corpus_id2user_id.pkl')
    corpus_user_ids = {corpus_id2user_id[i]: 0 for i in
                       xrange(len(corpus_id2user_id))}  # trick to make the loop faster

    valid_user_ids = Base.get_valid_user_ids(pref_data=pref_data, corpus_user_ids=corpus_user_ids,
                                             min_app_num=min_user_app_num)
    # user_profiles = Base.gen_user_profiles(valid_user_ids=valid_user_ids, user_vectors=user_vectors,
    #                                        corpus_id2user_id=corpus_id2user_id, user_data=user_data)

    user_ids = SArray(valid_user_ids)
    genders = []

    for user_id in valid_user_ids:
        if user_data[user_id]['gender'] == 'male':
            genders.append('male')
        elif user_data[user_id]['gender'] == 'female':
            genders.append('female')
        else:
            genders.append('None')

    genders = SArray(genders)

    # features = []
    # for feature_id in range(user_profiles.shape[1]):
    # for feature_id in range(1):
    #     features.append(SArray(user_profiles[:, feature_id]))

    # user_sf = {feature_id: features[feature_id] for feature_id in range(len(features))}
    user_sf = {'user_id': user_ids, 'gender': genders}
    # user_sf['user_id'] = user_ids
    user_sf = SFrame(user_sf)
    user_sf.save('data/converted/production/user_data.sf')


def gen_job_sframe():
    pass


if __name__ == '__main__':
    # logging
    logging = configure_log(__file__)
    #gen_pref_sframe()
    gen_user_sframe()
    # gen_job_sframe()
