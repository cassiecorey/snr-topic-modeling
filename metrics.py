from scipy.spatial.distance import cosine
from scipy.stats import entropy, signaltonoise
from numpy.linalg import norm

import math
import numpy as np

def coherence(topic):
    """

    """
    pass

# TODO: FIX THIS
def signal_to_noise(topic): 
    return signaltonoise(model)

def distance_from_corpus(doc_word, topic):
    # Total corpus word count
    S = np.sum(doc_word)
    corpus_distribution = np.array([x/S for x in np.sum(doc_word,axis=0)])
    distance = 0.0
    for i in range(len(topic)):
        a = topic[i]
        b = corpus_distribution[i]
        distance += a*ath.log(a/b)
    return distance

def distance_from_uniform(topic):
    distance = 0.0
    for p in topic:
        distance += p*math.log(p/1/len(topic))
    return distance

def exclusivity(topic_word,i,n=20):
    """
    Measures extent to which top words do not appear
    as top words in other topics.
    
    Average over each top word of:
    probability of that
    word in the topic divided by sum of probabilities
    of that word in all topics.
    """
    top_words_idx = topic_word[i].argsort()[:-n-1:-1]
    exclusivity = 0.0
    for j in top_words_idx:
        prob_topic_i = topic_word[i,j]
        prob_other_t = np.sum(topic_word[:,j])-prob_topic_i
        exclusivity += prob_topic_i/prob_other_t
    exclusivity /= n
    return exclusivity
        
def jensen_shannon_divergence(topic_a,topic_b):
    _topic_a = topic_a/norm(topic_a,ord=1)
    _topic_b = topic_b/norm(topic_b,ord=1)
    _m = 0.5*(_topic_a+topic_b)
    return 0.5*(entropy(_topic_a,_m)+entropy(_topic_b,_m))

def cosine_distance(topic_a,topic_b):
    return cosine(topic_a,topic_b)

def effective_size(topic):
    """
    From politics, effective size of parties.
    """
    size = 0.0
    for p in topic:
        size += math.pow(math.pow(p,2),-1)
    return size

def top_words(topic,feature_names,n=20):
    topic_sorted = topic.argsort()
    top_words = [feature_names[i] for i in topic_sorted[:-n-1:-1]]
    return top_words

def average_word_length(topic,feature_names,n=20):
    """
    Returns average length of top n words from topic.
    """
    tw = top_words(topic,feature_names,n)
    return np.array([len(w) for w in tw]).mean()

def rank1(doc_topic,i):
    """
    This is the likelihood of this topic being
    the most popular topic in a document. Calculated
    for the entire corpus.
    
    It's bad if this is high because it means your
    topic isn't special.
    """
    count=0
    for d in doc_topic:
        count += (np.argmax(d)==i)
    return count