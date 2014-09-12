"""
Executable file for LSA (Latent Semantic Analysis).

When running this file, 'working directory' need to be specified as Project Root (EnternshipsJobRecommendation).
"""
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from util.corpus import corpus2vector
from util.input import unpickle
from util.log import configure_log
from util.output import enpickle
import numpy as np

__author__ = 'kensk8er'


# def convert_dict_list(dict_data):
#     """
#     Convert dictionary format data into list format data. Return both list format data and indices that show which list
#     element corresponds to which dictionary element.
#
#     :param dict_data: dict[dict]
#     :return: list[str] list_data, list[str] indices
#     """
#     list_data = []
#     indices = []
#
#     for index, dict_datum in dict_data.items():
#         assert dict_datum.has_key('text'), 'dictionary data need to have the field \'text\'!'
#         assert isinstance(dict_datum['text'], str), 'str required!'
#         list_data.append(unicode(dict_datum['text'], 'utf-8'))  # convert str into unicode
#         indices.append(index)
#
#     return list_data, indices
#
#
# def calculate_similarities(document_matrix):
#     """
#     Calculate the similarities between every document vector which is contained in the document matrix given as an
#     argument.
#
#     :rtype : matrix[float]
#     :param document_matrix: Document Matrix whose each row contains a document vector.
#     """
#     # calculate inner products
#     print 'calculate inner products...'
#     inner_product_matrix = np.dot(document_matrix, document_matrix.T)
#
#     # calculate norms
#     print 'calculate norms...'
#     norms = np.sqrt(np.multiply(document_matrix, document_matrix).sum(1))
#     norm_matrix = np.dot(norms, norms.T)
#
#     # calculate similarities
#     print 'calculate similarities...'
#     similarity_matrix = inner_product_matrix / norm_matrix
#
#     return similarity_matrix


if __name__ == '__main__':
    # parameters
    tfidf = True
    num_topics = 5
    # chunksize = 2000

    # logging
    logging = configure_log(__file__)

    logging.info("variables: tfidf = %s, num_topics = %s" % (tfidf, num_topics))

    logging.info('Loading user dictionary...')
    user_dict = Dictionary.load('data/dictionary/user_(NN).dict')
    user_corpus = user_dict.corpus

    model_name = 'user_'

    if tfidf is True:
        logging.info('Computing TF-IDF...')
        tfidf_user = TfidfModel(user_corpus, normalize=False)
        logging.info('Transforming the user corpus...')
        user_corpus = [tfidf_user[corpus] for corpus in user_corpus]
        model_name += 'tfidf_'

    logging.info('Performing LSA on user corpus...')
    user_lsa = LsiModel(corpus=user_corpus, num_topics=num_topics, id2word=user_dict.id2token)
    user_lsa.save('data/feature/lsa/model/' + model_name + str(num_topics) + '.lsa')

    logging.info('Transforming the user corpus...')
    user_corpus = [user_lsa[corpus] for corpus in user_corpus]

    logging.info('Converting corpus into vector format...')
    user_vectors = corpus2vector(corpora=user_corpus, num_terms=len(user_dict.token2id))
    enpickle(user_vectors, 'data/feature/lsa/' + model_name + str(num_topics) + '.pkl')

    # TODO: implement LSA for job corpus

    # print 'read documents...'
    # documents = unpickle('data/txt/documents.pkl')
    # doc_num = len(documents)
    #
    # # convert dictionary format into list format
    # print 'convert dictionary into list format...'
    # doc_lists, doc_indices = convert_dict_list(documents)
    #
    # # Perform an IDF normalization on the output of HashingVectorizer
    # hasher = HashingVectorizer(stop_words='english', non_negative=True,
    #                            norm=None, binary=False)
    # vectorizer = Pipeline((
    #     ('hasher', hasher),
    #     ('tf_idf', TfidfTransformer())  # TODO: you should try many different parameters here
    # ))
    #
    # # reduce the number of documents for now
    # #doc_lists = doc_lists[:400]
    # #doc_indices = doc_indices[:400]
    #
    # # calculate TF-IDF
    # print 'calculate TF-IDF...'
    # X = vectorizer.fit_transform(doc_lists)
    #
    # # perform LSA
    # print 'perform LSA...'
    # lsa = TruncatedSVD(n_components=300, algorithm='arpack')
    # X = np.matrix(lsa.fit_transform(X))
    #
    # # calculate cosine similarities between each text
    # print 'calculate cosine similarities...'
    # similarities = calculate_similarities(X)
    #
    # print 'save similarities and indices...'
    # #date_time = datetime.datetime.today().strftime("%m%d%H%M%S")
    # enpickle(similarities, 'result/similarities.pkl')
    # enpickle(doc_indices, 'result/indices.pkl')

