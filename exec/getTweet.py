import csv
import re
from module.text.preprocessing import clean_text
from util.input import unpickle

__author__ = 'Linda Wang'

import twitter
import pandas as pd


if __name__ == '__main__':
    file_name = 'data/original/additional.csv'
    location_file = 'data/original/MSShopLocations.csv'

    api = twitter.Api(
        consumer_key='XrTlgFXqGt9XAnNFF7rec25wg',
        consumer_secret='heORHeyXvbTK7RozVLLeUjDrsOZWSg75eUjewLT6FXh8UuLAaz',
        access_token_key='2786813671-O1P8pWzrOAViJnZ0zkv3a4yl2rJHmbxcIfZUXub',
        access_token_secret='c0S9NKtzXU9dyRrsVqura0EIzXFBIWro3rvJBkrCdSyJe')

    print api.VerifyCredentials()
    # locations = pd.read_csv(location_file, quotechar='"')
    # locations['x'] = locations['Latitude/longitude'].str.split(',').apply(lambda x: x[0].strip())
    # locations['y'] = locations['Latitude/longitude'].str.split(',').apply(lambda x: x[1].strip())
    # del locations['Latitude/longitude']

    wikis = unpickle('data/others/wikis.pkl')
    keywords = wikis.keys()
    tweets = []
    # for i, row in enumerate(locations.iterrows()):
    for i, keyword in enumerate(keywords):
        try:
            print "{}th done".format(i)
            # res = api.GetSearch(geocode=(row[1].x, row[1].y, '0.1km'), count=100)
            res = api.GetSearch(term=keyword, lang='english', count=100)
            for tweet in res:
                # tweets.append(u'{},{},{}'.format(row[1].x, row[1].y, unicode(tweet.AsDict()['text']).replace('\n', '')))
                tweets.append(unicode(tweet.AsDict()['text']).replace('\n', ''))
        except Exception as e:
            print e

    f = open(file_name, 'ab')
    csvWriter = csv.writer(f)
    rx = re.compile('\W+')
    for tweet in tweets:
        tweet = clean_text(tweet)
        tweet = rx.sub(' ', tweet).strip()
        csvWriter.writerow([tweet])