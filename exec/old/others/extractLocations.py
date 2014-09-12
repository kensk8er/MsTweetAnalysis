from gensim.utils import lemmatize
import lxml.html
import re
from util.output import enpickle

__author__ = 'kensk8er'


if __name__ == '__main__':
    targetUrls = []
    # TODO: now only covering UK, expanding to the whole world
    doc = lxml.html.parse('http://en.wikipedia.org/wiki/List_of_United_Kingdom_locations')
    div = doc.xpath('//div[@id="mw-content-text"]')[0]
    uls = div.xpath('descendant::ul')

    for ul in uls:
        atags = ul.xpath('descendant::a')
        for atag in atags:
            pattern = re.compile('/wiki/List_of_United_Kingdom_locations:_[a-zA-Z-]+')
            if pattern.match(atag.attrib['href']):
                targetUrls.append('http://en.wikipedia.org' + atag.attrib['href'])

    locations = set()
    for targetUrl in targetUrls:
        doc = lxml.html.parse(targetUrl)
        div = doc.xpath('//div[@id="mw-content-text"]')[0]
        tables = div.xpath('descendant::table[@class="wikitable"]')
        for table in tables:
            atags = table.xpath('descendant::a[@class="new"]')
            for atag in atags:
                locations.add(atag.text)

            tds = table.xpath('descendant::td')
            for td in tds:
                if td.text is not None:
                    locations.add(td.text)

    location_terms = set()
    locations = [lemmatize(content=location, allowed_tags=re.compile('(NN)')) for location in locations if location is not None]
    for location in locations:
        for token in location:
            term, pos = token.split('/')
            location_terms.add(term)

    enpickle(location_terms, 'data/others/locations.pkl')
