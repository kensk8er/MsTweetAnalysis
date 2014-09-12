"""
Generate structured data of user-profiles, job-profiles, and preference (application-histories).
"""
from collections import defaultdict
from glob import glob
import logging
import os
from util.input import load_json, document_to_text, unpickle
from util.output import enpickle
from util.string import strip_suffix

__author__ = 'kensk8er'


def cvs2texts(dir_path, backup_path):
    """
    Load CVs and convert them into texts.

    :rtype : dict{str: str}
    :param dir_path:
    :return: dict of {app_id: text}
    """
    try:
        texts = unpickle('data/converted/production/user_texts.pkl')
    except:
        texts = {}
    texts = {}
    file_paths = glob(dir_path + '/' + '*.doc')  # FIXME: ugly implementation
    file_paths.extend(glob(dir_path + '/' + '*.docx'))
    file_paths.extend(glob(dir_path + '/' + '*.pdf'))
    file_paths.extend(glob(dir_path + '/' + '*.rtf'))

    for index, file_path in enumerate(file_paths):
        if index % 100 == 0:
            logging.info('%s / %s', str(index), str(len(file_paths)))

        if index % 1000 == 0:
            logging.info('Saving the backup..')
            enpickle(texts, 'data/converted/production/user_texts.pkl')

        dir_name, file_name = os.path.split(file_path)
        app_id = strip_suffix(string=file_name, suffix=['.doc', '.docx', '.pdf', '.rtf'])

        if app_id not in texts:  # try only when texts[app_id] isn't created yet
            try:
                text = document_to_text(file_path)
            except:
                logging.exception('Failed in converting the document into text.')
                continue

        texts[app_id] = text

    return texts


def gen_user_data(profiles, applications, texts):
    """


    :rtype : list[dict]
    :param profiles:
    :param applications:
    :param texts:
    :return: mapping from app_id (object id) to user_ids (object id)
    """

    def gen_userid2appids(applications):
        userid2appids = defaultdict(list)

        for application in applications:
            userid2appids[application['user_id']['$oid']].append(application['_id']['$oid'])
        return userid2appids


    user_data = []
    userid2appids = gen_userid2appids(applications)

    for profile in profiles:
        user_id = profile['user_id']['$oid']
        user_datum = {'id': len(user_data), 'user_id': user_id, 'facebook_url': profile['facebook_url'],
                      'gender': profile['gender'], 'linkedin_url': profile['linkedin_url'],
                      'twitter_url': profile['twitter_url'], 'website_url': profile['website_url'], 'cv': None}

        if userid2appids.has_key(user_id):
            # FIXME: This implementation always use the first CV, and discard others.
            app_id = userid2appids[user_id][0]

            if texts.has_key(app_id):
                user_datum['cv'] = texts[app_id]

        user_data.append(user_datum)

    return user_data


def gen_job_data(companies, functions, industries, listings, locations, skills):
    """


    :rtype : list[dict]
    :param companies:
    :param functions:
    :param industries:
    :param listings:
    :param locations:
    :param skills:
    :return:
    """

    def gen_compid2company(companies):
        compid2company = {}
        for company in companies:
            compid2company[company['_id']['$oid']] = company
        return compid2company


    def gen_funcid2funcname(functions):
        funcid2funcname = {}
        for function in functions:
            funcid2funcname[function['_id']['$oid']] = function['name']
        return funcid2funcname


    def gen_skillid2skillname(skills):
        skillid2skillname = {}
        for skill in skills:
            skillid2skillname[skill['_id']['$oid']] = skill['name']
        return skillid2skillname


    def gen_industryid2industry(industries):
        industryid2industry = {}
        for industry in industries:
            industryid2industry[industry['_id']['$oid']] = industry['name']
        return industryid2industry


    job_data = []
    # FIXME: ugly implementation
    compid2company = gen_compid2company(companies)
    funcid2funcname = gen_funcid2funcname(functions)
    industryid2industry = gen_industryid2industry(industries)
    skillid2skillname = gen_skillid2skillname(skills)

    for listing in listings:
        listing['oid'] = listing['_id']['$oid']
        listing['id'] = len(job_data)
        listing['company'] = compid2company[listing['company_id']['$oid']] if compid2company.has_key(
            listing['company_id']['$oid']) else None
        listing['function_name'] = funcid2funcname[listing['function_id']['$oid']]
        listing['industry_name'] = industryid2industry[listing['industry_id']['$oid']]
        if listing.has_key('skill_ids'):
            listing['skill_names'] = []
            for skill_id in listing['skill_ids']:
                if skillid2skillname.has_key(skill_id['$oid']):
                    listing['skill_names'].append(skillid2skillname[skill_id['$oid']])
            if len(listing['skill_names']) == 0:
                listing['skill_names'] = None
        else:
            listing['skill_names'] = None
        job_data.append(listing)

    return job_data


def gen_pref_data(applications, user_data, job_data):
    """


    :rtype : list[list]
    :param applications:
    :param user_data:
    :param job_data:
    :return:
    """

    def gen_useroid2userid(user_data):
        useroid2userid = {}
        for user_datum in user_data:
            useroid2userid[user_datum['user_id']] = user_datum['id']
        return useroid2userid

    def gen_listingid2jobid(job_data):
        listingid2jobid = {}
        for job_datum in job_data:
            listingid2jobid[job_datum['oid']] = job_datum['id']
        return listingid2jobid

    pref_data = [[0 for i in xrange(len(job_data))] for j in xrange(len(user_data))]  # initialize
    useroid2userid = gen_useroid2userid(user_data)
    listingid2jobid = gen_listingid2jobid(job_data)

    for application in applications:
        useroid = application['user_id']['$oid']
        listingid = application['listing_id']['$oid']
        if useroid2userid.has_key(useroid) and listingid2jobid.has_key(listingid):  # some listings don't exist
            pref_data[useroid2userid[useroid]][listingid2jobid[listingid]] = 1

    return pref_data


if __name__ == '__main__':
    # variables
    json_dir_path = 'data/original/production'
    cv_dir_path = '/Users/kensk8er/Google Drive (kensk8er)/kensuke_muraki_cvs (1)'

    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logging.info('Loading json files...')
    json_data = load_json(dir_path=json_dir_path)

    logging.info('Converting CV data into texts...')
    user_texts = cvs2texts(dir_path=cv_dir_path, backup_path='data/converted/production/user_texts.pkl')
    enpickle(user_texts, 'data/converted/production/user_texts.pkl')

    logging.info('Generating user data structure...')
    user_data = gen_user_data(profiles=json_data['profiles'], applications=json_data['applications'], texts=user_texts)
    enpickle(user_data, 'data/converted/production/user_data.pkl')

    logging.info('Generating job data structure...')
    job_data = gen_job_data(companies=json_data['companies'], functions=json_data['functions'],
                            industries=json_data['industries'], listings=json_data['listings'],
                            locations=json_data['locations'], skills=json_data['skills'])
    enpickle(job_data, 'data/converted/production/job_data.pkl')

    logging.info('Generating preference data structure...')
    pref_data = gen_pref_data(applications=json_data['applications'], user_data=user_data, job_data=job_data)
    enpickle(pref_data, 'data/converted/production/pref_data.pkl')
