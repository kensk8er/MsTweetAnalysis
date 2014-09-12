"""
Computing the similarities between job posts.
"""
from collections import defaultdict
from util.input import unpickle
from util.log import configure_log
import numpy as np
from util.output import enpickle

__author__ = 'kensk8er'


if __name__ == '__main__':
    # logging
    logging = configure_log(__file__)

    logging.info('Loading job data...')
    job_data = unpickle('data/converted/production/job_data.pkl')
    valid_job_ids = np.array(range(len(job_data)))

    job_id_converter = {}
    job_index = defaultdict(dict)
    duplicate_count = 0

    logging.info('Checking duplicate jobs...')
    for job_datum in job_data:
        job_name = job_datum['name']
        industry = job_datum['industry_name']
        company_name = job_datum['company']['name'] if job_datum['company'] is not None else None
        key = (job_name, industry, company_name)

        if job_index.has_key(key):
            job_id_converter[job_datum['id']] = job_index[key]['id']
            job_index[key]['count'] += 1
            duplicate_count += 1
        else:
            job_index[key]['id'] = job_datum['id']
            job_index[key]['count'] = 1

    logging.info("job post num: %s" % len(valid_job_ids))
    logging.info("duplicate num: %s" % duplicate_count)
    enpickle(job_id_converter, 'data/converted/production/job_id_converter.pkl')
    enpickle(job_index, 'data/converted/production/job_duplicate_index.pkl')

    logging.info('Converting preference data...')
    pref_data = unpickle('data/converted/production/pref_data_sparse.pkl')
    pref_data = pref_data.tolil()

    for user_id in xrange(pref_data.shape[0]):
        print "\r%s / %s" % (user_id + 1, pref_data.shape[0]),
        job_ids = pref_data.getrow(user_id).nonzero()[1]
        for job_id in job_ids:
            if job_id_converter.has_key(job_id):
                pref_data[user_id, job_id] = 0
                pref_data[user_id, job_id_converter[job_id]] = 1

    pref_data = pref_data.tocoo()
    enpickle(pref_data, 'data/converted/production/pref_data_converted.pkl')