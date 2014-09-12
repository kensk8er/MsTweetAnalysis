from gensim.utils import lemmatize
import lxml.html
import re
from util.output import enpickle

__author__ = 'kensk8er'


if __name__ == '__main__':
    names = []
    doc = lxml.html.parse('http://simple.wikiquote.org/wiki/List_of_people_by_name')
    div = doc.xpath('//div[@id="mw-content-text"]')[0]
    uls = div.xpath('descendant::ul')

    for ul in uls:
        atags = ul.xpath('descendant::a')
        for atag in atags:
            names.append(atag.text)

    name_terms = set()
    names = [lemmatize(content=name, allowed_tags=re.compile('(NN)')) for name in names if name is not None]
    for name in names:
        for token in name:
            term, pos = token.split('/')
            name_terms.add(term)

    enpickle(name_terms, 'data/others/names.pkl')
