"""
Metrics Topic Model
"""

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy
from numpy.linalg import norm
from scipy.spatial.distance import cosine

import numpy as np
import math


def from_samples(data_samples,n_topics=10,document_topic=None,topic_word=None):
    """
    Builds a MetricsModel from a list of raw documents.
    """
    

class MetricsModel():
    """
    Topics model that knows its metrics.
    """

    def __init__(self, n_topics=None,
                 topic_word=None, document_topic=None,
                 document_word=None, features=None):
        self.n_topics=n_topics
        self.topic_word=topic_word
        self.document_topic=document_topic
        self.document_word=document_word
        self.features=features

    def fit_from_samples(self, data_samples,n_topics,
                         doc_topic_prior=None,topic_word_prior=None):
        """
        Fits the model from a raw sample of text documents.

        Currently uses LDA to generate the matrices.
        """
        # TODO: MAKE ALPHA ASYMMETRIC
        lda = LatentDirichletAllocation(n_components=n_topics,
                                        doc_topic_prior=doc_topic_prior,
                                        topic_word_prior=topic_word_prior,
                                        learning_method='online',
                                        random_state=0)

        # Converts a list of documents into their word-counts
        tf_vectorizer = CountVectorizer(stop_words='english')
        tf = tf_vectorizer.fit_transform(data_samples)
        lda.fit(tf)
        
        self.n_topics=n_topics
        self.topic_word=lda.components_
        self.document_topic=lda.transform(tf)
        self.document_word=tf.toarray()
        self.features=tf_vectorizer.get_feature_names()

    def generate(self,n_documents=100,n_words=100,document_topic_prior=None):
        """
        Generate document_word.
        Optionally, you can specify a document_topic or topic_word priors.
        If you include this prior it should agree in shape with what you
        put for n_documents.
        """
        doc_word = [[0 for i in range(n_words)] for d in range(n_documents)]
        for d in range(n_documents):   
            doc = []
            # Choose one topic for the document.
            # Not exactly LDA protocol...         
            if document_topic_prior is not None:
                t = np.random.choice(range(self.n_topics),p=document_topic_prior[d])
            else:
                t = np.random.choice(range(self.n_topics))
            topic = self.topic_word[t]
            # Make n_words selections from the topic distributions
            w = np.random.choice(range(len(topic)),p=topic)
            doc_word[d][w]+=1
        return doc_word

    def top_words(self,i,n_top_words=10):
        topic = self.topic_word[i]
        sorted_top = topic.argsort()[:-n_top_words-1:-1]
        return [self.features[w] for w in sorted_top]
        
    def print_top_words(self,i,n_top_words=10):
        top_words = top_words(model_components,n_top_words)
        for topic in top_words:
            words = ' '.join(top_words[topic])
            print("Topic_{}: {}".format(topic,words))

    def coherence(self,i):
        """
        TODO: Select a coherence algorithm & run.
        """
        pass

    def signal_to_noise(self,i):
        """
        Calculate the signal to noise ratio.

        TODO: Fix this.
        """
        topic = self.topic_word[i]
        return signaltonoise(topic)

    def distance_from_corpus(self,i):
        """
        Measures distance from corpus distribution.
        Distance is measured as Kullback-Leibler divergence.
        """
        doc_word = self.document_word
        topic = self.topic_word[i]
        # Total corpus word count (minus stopwords)
        S = doc_word.sum()
        corpus_distribution = doc_word.sum(axis=0)/S
        distance = 0.0
        for i in range(len(topic)):
            a = topic[i] # word i's distribution in topic
            b = corpus_distribution[i] # word i's distribution in corpus
            distance += a*math.log(a/b)
        return distance

    def distance_from_uniform(self,i):
        """
        Measures distance from a uniform distribution
        over words in the vocabulary.
        Distance is measured as Kullback-Leibler divergence.
        """
        topic = self.topic_word[i]
        distance = 0.0
        for p in topic:
            distance += p*math.log(p/(1/len(topic)))
        return distance

    def exclusivity(self,i,n=20):
        """
        Measures extent to which top words do not appear
        as top words in other topics.
        Average over each top word of the probability of
        that word in the topic divided by sum of
        probabilities of that word in all topics.
        """
        topic = self.topic_word[i]
        topic_word = self.topic_word

        top_words_idx = topic_word[i].argsort()[:-n-1:-1]
        exclusivity = 0.0
        for j in top_words_idx:
            prob_topic_i = topic_word[i,j]
            prob_other_t = np.sum(topic_word[:,j])-prob_topic_i
            exclusivity += prob_topic_i/prob_other_t
        exclusivity /= n
        return exclusivity
            
    def effective_size(self,i):
        """
        From politics, effective size of parties.
        """
        topic = self.topic_word[i]
        size = 0.0
        for p in topic:
            size += math.pow(math.pow(p,2),-1)
        return size

    def top_words(self,i,n=20):
        """
        Get the top N most-likely words from a topic.
        """
        topic = self.topic_word[i]
        feature_names = self.features
        topic_sorted = topic.argsort()
        top_words = [feature_names[i] for i in topic_sorted[:-n-1:-1]]
        return top_words

    def average_word_length(self,i,n=20):
        """
        Returns average length of top n words from topic.
        """
        tw = top_words(i,n)
        return np.array([len(w) for w in tw]).mean()

    def rank1(self,i):
        """
        This is the likelihood of this topic being
        the most popular topic in a document. Calculated
        for the entire corpus.
        
        It's bad if this is high because it means your
        topic isn't special.
        """
        doc_topic = self.document_topic
        count=0
        for d in doc_topic:
            count += (np.argmax(d)==i)
        return count

    def avg_kld(self,i):
        all_klds = []
        for j in range(self.n_topics):
            all_klds.append(self.kullback_leibler_divergence(i,j))
        return np.sum(all_klds)/len(all_klds)

    def avg_jsd(self,i):
        all_jsds = []
        for j in range(self.n_topics):
            all_jsds.append(self.jensen_shannon_divergence(i,j))
        return np.sum(all_jsds)/len(all_jsds)

    def avg_cos(self,i):
        all_cos = []
        for j in range(self.n_topics):
            all_cos.append(self.cosine_distance(i,j))
        return np.sum(all_cos)/len(all_cos)

    ######################################
    # MULTI-TOPIC and FULL MODEL METRICS #
    ######################################

    def kullback_leibler_divergence(self,i,j):
        """
        Kives KLD between two topics.
        Input is the indices of the topics.
        """
        u = self.topic_word[i]
        v = self.topic_word[j]
        _u = u/norm(u,ord=1)
        _v = v/norm(v,ord=1)
        _m = 0.5*(_u+_v)
        return entropy(u,_m)

    def jensen_shannon_divergence(self,i,j):
        """
        Gives JSD between two topics.
        Input is the indices of the topics.
        """
        u = self.topic_word[i]
        v = self.topic_word[j]
        _u = u/norm(u,ord=1)
        _v = v/norm(v,ord=1)
        _m = 0.5*(_u+_v)
        return 0.5*(entropy(_u,_m)+entropy(_v,_m))

    def cosine_distance(self,i,j):
        """
        Gives cosine distance between two topics.
        Input is the indices of the topics.
        """
        u = self.topic_word[i]
        v = self.topic_word[j]
        return cosine(u,v)
