"""

"""
import csv
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from util.input import unpickle

__author__ = 'kensk8er'


def write_topics(model_path, csv_name, k):
    model = LdaModel.load(model_path)
    topics = []
    for topic_id in range(model.num_topics):
        topics.append(model.return_topic(topicid=topic_id))

    dictionary = Dictionary.load('data/dictionary/tweets.dict')
    word_indices = dictionary.id2token
    writer = csv.writer(file(csv_name, 'w'))

    output = [[0 for i in range(model.num_topics)] for j in range(k)]
    for topic_id, topic in enumerate(topics):
        for rank, index in enumerate(topic.argsort()[::-1]):
            output[rank][topic_id] = {}
            output[rank][topic_id]['word'] = word_indices[index]
            output[rank][topic_id]['p'] = topic[index]
            rank += 1
            if rank >= k:
                break

    for topic_id in range(model.num_topics):
        row = ['z = ' + str(topic_id)]

        for rank in range(k):
            row.append(output[rank][topic_id]['word'] + ':' + str(output[rank][topic_id]['p']))

        writer.writerow(row)


def write_doc_topics(vector_path, id_path, csv_name):
    vectors = unpickle(vector_path)
    ids = unpickle(id_path)
    writer = csv.writer(file(csv_name, 'w'))

    # 1st row
    row = ['']
    for topic_id in range(vectors.shape[1]):
        row.append('z = ' + str(topic_id))
    writer.writerow(row)

    # 2nd row and onwards
    for row_num, id in enumerate(ids):
        row = [id]

        for topic_id in range(vectors.shape[1]):
            row.append(vectors[row_num, topic_id])

        writer.writerow(row)


if __name__ == '__main__':
    write_topics(model_path='data/model/tweets_wiki_100.lda', csv_name='result/tweets_wiki_100_topic.csv', k=100)
    write_doc_topics(vector_path='data/vector/tweets_wiki_100.pkl', id_path='data/vector/ids.pkl',
                     csv_name='result/tweets_wiki_100_doc.csv')