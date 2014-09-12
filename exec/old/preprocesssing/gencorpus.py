"""
Generate corpus for both user and job data.
"""
import logging
import re
from util.input import unpickle
from util.output import enpickle

__author__ = 'kensk8er'


def gen_user_corpora(user_data, min_len=1000):
    """


    :rtype : dict[id: text]
    :param user_data:
    :return:
    """
    user_corpora = {}
    for user_datum in user_data:
        # TODO: currently only considering CV data, also incorporating LinkedIn profile next.
        id = user_datum['id']
        user_corpora[id] = user_datum['cv']

        # discard the corpus below the minimum length
        if user_corpora[id] is not None and len(user_corpora[id]) < min_len:
            user_corpora[id] = None

    return user_corpora


def gen_job_corpora(job_data, title_weight=3):  # TODO: examine title_weight
    """


    :type job_data: dict[id: text]
    :param job_data:
    :return:
    """
    # TODO: treat different section as different features
    job_corpora = {}
    for job_datum in job_data:
        id = job_datum['id']
        job_corpora[id] = unicode()

        # FIXME: inappropriate exception handling and ugly implementation
        job_corpora[id] += (job_datum['name'] + ' ') * title_weight
        try:
            job_corpora[id] += job_datum['company']['description'] + ' '
        except:
            pass
        try:
            job_corpora[id] += job_datum['outcomes'] + ' '
        except:
            pass

        # remove html-tags from job description and job requirements # TODO: utilize tag structure
        htmltag = re.compile(r'<.*?>', re.I | re.S)
        try:
            job_corpora[id] += htmltag.sub('', job_datum['description']) + ' '
        except:
            pass
        try:
            job_corpora[id] += htmltag.sub('', job_datum['requirements']) + ' '
        except:
            pass

    return job_corpora


if __name__ == '__main__':
    # variables
    user_data_path = 'data/converted/production/user_data.pkl'
    job_data_path = 'data/converted/production/job_data.pkl'

    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    logging.info('Loading user data...')
    user_data = unpickle(user_data_path)

    logging.info('Generating user corpora...')
    user_corpora = gen_user_corpora(user_data)
    enpickle(user_corpora, 'data/rawcorpus/user_corpora.pkl')

    logging.info('Loading user data...')
    job_data = unpickle(job_data_path)

    logging.info('Generating job corpora...')
    job_corpora = gen_job_corpora(job_data)
    enpickle(job_corpora, 'data/rawcorpus/job_corpora.pkl')
