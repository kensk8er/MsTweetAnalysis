"""
Executable file to generate wikipedia corpus.
"""
import logging
from gensim.utils import lemmatize, revdict
import pattern
from pattern.web import Wikipedia
import re
from module.text.preprocessing import clean_text, convert_compound
from module.text.stopword import stopwords
from module.wiki.topics import get_keywords
from util.input import unpickle
from util.output import enpickle

__author__ = 'kensk8er'


def crawl_wiki(model_path):
    engine = Wikipedia(license=None, throttle=1.0, language='en')
    wikis = {}
    keywords = get_keywords(model_path=model_path, threshold=0.001)
    for keyword in keywords:
        stop = False
        while stop is False:
            try:
                article = engine.search(query=keyword)
            except Exception as e:
                print str(e)
                article = None

            if type(article) is pattern.web.WikipediaArticle:
                if article.disambiguation is False:
                    print '\nretrieving', keyword, '...',
                    wikis[keyword] = {}
                    wikis[keyword]['keyword'] = keyword
                    wikis[keyword]['text'] = article.plaintext()
                    stop = True
                else:
                    print '\n[', keyword, '] leads to disambiguation page!',
                    stop = True

                    if '-' in keyword:
                        keyword = re.sub('-', ' ', keyword)  # convert hyphen into white space
                        stop = False
                    if keyword.islower() and len(keyword) <= 5:
                        keyword = keyword.upper()
                        stop = False
            else:
                print '\n[', keyword, '] doesn\'t exist on wikipedia!',
                stop = True

                if '-' in keyword:
                    keyword = re.sub('-', ' ', keyword)  # convert hyphen into white space
                    stop = False
                if keyword.islower() and len(keyword) <= 5:
                    keyword = keyword.upper()
                    stop = False

    enpickle(wikis, 'data/others/wikis.pkl')
    print '\n'
    return wikis


if __name__ == '__main__':
    # hyper-parameters
    allowed_pos = re.compile('(NN)')
    crawl = True
    # target = 'user'
    # topic_num = 100
    model_path = 'data/model/tweets_100.lda'

    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    if crawl is True:
        logging.info('Crawling wikipedia...')
        wikis = crawl_wiki(model_path=model_path)
    else:
        wikis = unpickle('data/others/wikis.pkl')

    logging.info('Lemmatizing wikipedia texts...')
    count = 0
    doc_num = len(wikis)
    new_wikis = []
    keywords = []
    for keyword, wiki in wikis.items():
        count += 1

        print '\r', count, '/', doc_num,
        text = wiki['text']
        cleaned = clean_text(text)  # delete irrelevant characters

        wiki = []
        tokens = lemmatize(content=cleaned, allowed_tags=allowed_pos)  # lemmatize
        for token in tokens:
            word, pos = token.split('/')
            wiki.append(word)

        # convert compound word into one token
        wiki = convert_compound(wiki)

        # filter stop words, long words, and non-english words
        wiki = [w for w in wiki if not w in stopwords and 2 <= len(w) <= 15 and w.islower()]  # FIXME: this allows non-english characters to be stored

        new_wikis.append(wiki)
        keywords.append(keyword)

    print '\n'

    logging.info('Saving wiki corpus...')
    enpickle(new_wikis, 'data/processed/wikis.pkl')
