"""
Specify topics to extract knowledge from wikipedia.
"""
from gensim.models import LdaModel
import numpy as np

__author__ = 'kensk8er'


def get_keywords(model_path, threshold=0.01):
    lda_model = LdaModel.load(model_path)
    topic_num = lda_model.num_topics
    keywords = set()
    for topic_id in range(topic_num):
        topic = lda_model.state.get_lambda()[topic_id]
        topic = topic / topic.sum()  # normalize to probability dist
        signif_word_ids = np.where(topic > threshold)[0]
        keywords = keywords.union([lda_model.id2word[word_id] for word_id in signif_word_ids])

    return keywords
