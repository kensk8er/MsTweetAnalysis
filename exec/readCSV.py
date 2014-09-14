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

    ms_hash = {}
    text_index = {}
    for row in ms_hash_csv:
        id, text = row
        if not text_index.has_key(text):
            text_index[text] = ''
            ms_hash[id] = text

    enpickle(ms_hash, 'data/processed/ms_hash.pkl')

    # M&S location
    ms_loc_file = open('data/original/msgsdeduped.csv', 'rb')
    ms_loc_csv = csv.reader(ms_loc_file)

    ms_loc = {}
    text_index = {}
    for row in ms_loc_csv:
        id, text = row
        if not text_index.has_key(text):
            text_index[text] = ''
            ms_loc[id] = text

    enpickle(ms_loc, 'data/processed/ms_loc.pkl')

    # other english tweets
    other_file = open('data/original/agreegatedEnglishTweets.csv', 'rb')
    other_csv = csv.reader(other_file)

    other = {}
    text_index = {}
    for row in other_csv:
        id, text = row
        if not text_index.has_key(text):
            text_index[text] = ''
            other[id] = text

    enpickle(other, 'data/processed/other.pkl')


