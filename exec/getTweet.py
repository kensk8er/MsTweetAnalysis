import csv
import re
from time import sleep
from module.text.preprocessing import clean_text
from module.wiki.topics import get_keywords
import twitter

__author__ = 'Linda Wang'

if __name__ == '__main__':
    file_name = 'data/original/additional.csv'
    location_file = 'data/original/MSShopLocations.csv'

    api = twitter.Api(
        consumer_key='XrTlgFXqGt9XAnNFF7rec25wg',
        consumer_secret='heORHeyXvbTK7RozVLLeUjDrsOZWSg75eUjewLT6FXh8UuLAaz',
        access_token_key='2786813671-O1P8pWzrOAViJnZ0zkv3a4yl2rJHmbxcIfZUXub',
        access_token_secret='c0S9NKtzXU9dyRrsVqura0EIzXFBIWro3rvJBkrCdSyJe')

    print api.VerifyCredentials()

    # wikis = unpickle('data/others/wikis.pkl')
    # keywords = wikis.keys()
    keywords = get_keywords(model_path='data/model/tweets_100.lda')
    tweets = []
    for i, keyword in enumerate(keywords):
        try:
            print "{}th done".format(i)
            res = api.GetSearch(term=keyword, lang='english', count=100)
            for tweet in res:
                tweets.append(unicode(tweet.AsDict()['text']).replace('\n', ''))
        except Exception as e:
            print e

        if i % 150 == 0 and i > 0:
            sleep(60*15)  # avoid exceeding the API limit

    f = open(file_name, 'ab')
    csvWriter = csv.writer(f)
    rx = re.compile('\W+')
    for tweet in tweets:
        tweet = clean_text(tweet)
        tweet = rx.sub(' ', tweet).strip()
        csvWriter.writerow([tweet])