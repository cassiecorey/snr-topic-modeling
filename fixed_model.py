# uncompyle6 version 3.2.3
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.1 |Anaconda custom (x86_64)| (default, Dec  7 2015, 11:24:55) 
# [GCC 4.2.1 (Apple Inc. build 5577)]
# Embedded file name: /Users/cassiancorey/Google Drive/Fall '17/691DD/project/fixed_model.py
# Compiled at: 2017-11-13 18:12:04
# Size of source mod 2**32: 2149 bytes
"""
Constructs a baseline model for topic-modeling.
"""
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
K = 30
ALPHA = 0.06666666666666667
BETA = 1 / K
lda = LatentDirichletAllocation(n_components=K, doc_topic_prior=ALPHA, topic_word_prior=BETA, learning_method='online', random_state=0)
tf_vectorizer = CountVectorizer(stop_words='english')

def get_model(data_samples):
    global lda
    model_components = {}
    tf = tf_vectorizer.fit_transform(data_samples)
    lda.fit(tf)
    model_components['features'] = tf_vectorizer.get_feature_names()
    model_components['topic_word'] = lda.components_
    model_components['doc_word'] = tf.toarray()
    model_components['doc_topic'] = lda.transform(tf)
    return model_components


def update_model(num_topics, doc_topic_prior=None, topic_word_prior=None):
    global ALPHA
    global BETA
    global lda
    if doc_topic_prior is not None:
        ALPHA = doc_topic_prior
    if topic_word_prior is not None:
        BETA = topic_word_prior
    lda = LatentDirichletAllocation(n_components=K, doc_topic_prior=ALPHA, topic_word_prior=BETA, learning_method='online')


def get_top_words(model_components, n_top_words=10):
    top_words = {}
    for topic_idx, topic in enumerate(model_components['topic_word']):
        sorted_top = topic.argsort()[:-n_top_words - 1:-1]
        top_words[topic_idx] = [model_components['features'][i] for i in sorted_top]

    return top_words


def print_top_words(model_components, n_top_words=10):
    top_words = get_top_words(model_components, n_top_words)
    for topic in top_words:
        words = (' ').join(top_words[topic])
        print(('Topic_{}: {}').format(topic, words))