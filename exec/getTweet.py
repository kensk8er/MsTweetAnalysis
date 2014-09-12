__author__ = 'Linda Wang'

import twitter
import pandas as pd

file_name = 'data/additional.csv'

api = twitter.Api(
    consumer_key='XrTlgFXqGt9XAnNFF7rec25wg',
    consumer_secret='heORHeyXvbTK7RozVLLeUjDrsOZWSg75eUjewLT6FXh8UuLAaz',
    access_token_key='2786813671-O1P8pWzrOAViJnZ0zkv3a4yl2rJHmbxcIfZUXub',
    access_token_secret='c0S9NKtzXU9dyRrsVqura0EIzXFBIWro3rvJBkrCdSyJe')

print api.VerifyCredentials()
locations = pd.read_csv('/Users/lindawang/Desktop/sentiment', quotechar='"')
locations['x'] = locations['Latitude/longitude'].str.split(',').apply(lambda x: x[0].strip())
locations['y'] = locations['Latitude/longitude'].str.split(',').apply(lambda x: x[1].strip())
del locations['Latitude/longitude']

tweets = []
for i, row in enumerate(locations.iterrows()):
    try:
        print "{}th done".format(i)
        res = api.GetSearch(geocode=(row[1].x, row[1].y, '0.1km'), count=100)
        for tweet in res:
            tweets.append(u'{},{},{}'.format(row[1].x, row[1].y, unicode(tweet.AsDict()['text']).replace('\n', '')))
    except Exception as e:
        print e

with open(file_name, 'w+') as sink:
    sink.write(u'\n'.join(tweets))

