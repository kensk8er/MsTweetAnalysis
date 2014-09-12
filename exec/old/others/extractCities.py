from gensim.utils import lemmatize
import lxml.html
import re
from util.output import enpickle

__author__ = 'kensk8er'


if __name__ == '__main__':
    cities = set()
    # TODO: now only covering UK, expanding to the whole world
    doc = lxml.html.parse('http://en.wikipedia.org/wiki/Global_city')
    div = doc.xpath('//div[@id="mw-content-text"]')[0]
    tables = div.xpath('descendant::table[@class="wikitable plainrowheaders"]')

    for table in tables:
        atags = table.xpath('descendant::a')
        for atag in atags:
            if atag.text is not None:
                cities.add(atag.text)

    tables = div.xpath('descendant::table[@class="wikitable"]')

    for table in tables:
        atags = table.xpath('descendant::a')
        for atag in atags:
            if atag.text is not None:
                cities.add(atag.text)

    city_terms = set()
    cities = [lemmatize(content=city, allowed_tags=re.compile('(NN)')) for city in cities if city is not None]
    for city in cities:
        for token in city:
            term, pos = token.split('/')
            city_terms.add(term)

    enpickle(city_terms, 'data/others/cities.pkl')
