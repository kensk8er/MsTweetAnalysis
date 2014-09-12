from gensim.utils import lemmatize
import lxml.html
import re
from util.output import enpickle

__author__ = 'kensk8er'


if __name__ == '__main__':
    schools = []
    # TODO: now only covering UK, expanding to the whole world
    doc = lxml.html.parse('http://en.wikipedia.org/wiki/List_of_universities_in_the_United_Kingdom')
    div = doc.xpath('//div[@id="mw-content-text"]')[0]
    uls = div.xpath('descendant::ul')

    for ul in uls:
        atags = ul.xpath('descendant::a')
        for atag in atags:
            schools.append(atag.text)

    school_terms = set()
    schools = [lemmatize(content=school, allowed_tags=re.compile('(NN)')) for school in schools if school is not None]
    for school in schools:
        for token in school:
            term, pos = token.split('/')
            school_terms.add(term)

    enpickle(school_terms, 'data/others/schools.pkl')
