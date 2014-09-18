"""
Reading a CSV file and store it in .pkl file.
"""
import csv
from util.output import enpickle

__author__ = 'kensk8er'


if __name__ == '__main__':
    # M&S mention and hash-tag
    ms_hash_file = open('data/original/aggregatedm&s.csv', 'rb')
    ms_hash_csv = csv.reader(ms_hash_file)
    tweets = {}
    text_index = {}
    for row in ms_hash_csv:
        id, text = row
        id = 'ms_hash_' + id
        if len(text) < 10:
            continue
        if not text_index.has_key(text):
            text_index[text] = ''
            tweets[id] = text

    # M&S location
    ms_loc_file = open('data/original/msgsdeduped.csv', 'rb')
    ms_loc_csv = csv.reader(ms_loc_file)
    for row in ms_loc_csv:
        id, text = row
        id = 'ms_loc_' + id
        if len(text) < 10:
            continue
        if not text_index.has_key(text):
            text_index[text] = ''
            tweets[id] = text

    # other english tweets
    other_file = open('data/original/agreegatedEnglishTweets.csv', 'rb')
    other_csv = csv.reader(other_file)
    for row in other_csv:
        id, text = row
        id = 'other_' + id
        if len(text) < 10:
            continue
        if not text_index.has_key(text):
            text_index[text] = ''
            tweets[id] = text

    # additional tweets
    other_file = open('data/original/additional.csv', 'rb')
    other_csv = csv.reader(other_file)
    for row in other_csv:
        id, text = row
        id = 'additional_' + id
        if len(text) < 10:
            continue
        if not text_index.has_key(text):
            text_index[text] = ''
            tweets[id] = text

    enpickle(tweets, 'data/processed/tweets.pkl')