from scipy.spatial.distance import cosine
from scipy.stats import entropy, signaltonoise
from numpy.linalg import norm

import math
import numpy as np

############################
# INDIVIDUAL TOPIC METRICS #
############################

def coherence(topic):
    """
    TODO: Select a coherence algorithm & run.
    """
    pass

def signal_to_noise(model_components,i):
    """
    Calculate the signal to noise ratio.

    TODO: Fix this.
    """
    topic = model_components['topic_word'][i]
    return signaltonoise(topic)

def distance_from_corpus(model_components,i):
    """
    Measures distance from corpus distribution.
    """
    doc_word = model_components['doc_word']
    topic = model_components['topic_word'][i]
    # Total corpus word count
    S = np.sum(doc_word)
    corpus_distribution = np.array([x/S for x in np.sum(doc_word,axis=0)])
    distance = 0.0
    for i in range(len(topic)):
        a = topic[i]
        b = corpus_distribution[i]
        distance += a*ath.log(a/b)
    return distance

def distance_from_uniform(model_components,i):
    """
    Measures distance from a uniform distribution
    over words in the vocabulary.
    """
    topic = model_components['topic_word'][i]
    distance = 0.0
    for p in topic:
        distance += p*math.log(p/1/len(topic))
    return distance

def exclusivity(model_components,i,n=20):
    """
    Measures extent to which top words do not appear
    as top words in other topics.
    
    Average over each top word of the probability of
    that word in the topic divided by sum of
    probabilities of that word in all topics.
    """
    topic = model_components['topic_word'][i]
    topic_word = model_components['topic_word']

    top_words_idx = topic_word[i].argsort()[:-n-1:-1]
    exclusivity = 0.0
    for j in top_words_idx:
        prob_topic_i = topic_word[i,j]
        prob_other_t = np.sum(topic_word[:,j])-prob_topic_i
        exclusivity += prob_topic_i/prob_other_t
    exclusivity /= n
    return exclusivity
        
def effective_size(topic):
    """
    From politics, effective size of parties.
    """
    size = 0.0
    for p in topic:
        size += math.pow(math.pow(p,2),-1)
    return size

def top_words(model_components,i,n=20):
    """
    Get the top N most-likely words from a topic.
    """
    topic = model_components['topic_word'][i]
    feature_names = model_components['features']
    topic_sorted = topic.argsort()
    top_words = [feature_names[i] for i in topic_sorted[:-n-1:-1]]
    return top_words

def average_word_length(model_components,i,n=20):
    """
    Returns average length of top n words from topic.
    """
    tw = top_words(model_components,i,n)
    return np.array([len(w) for w in tw]).mean()

def rank1(model_components ,i):
    """
    This is the likelihood of this topic being
    the most popular topic in a document. Calculated
    for the entire corpus.
    
    It's bad if this is high because it means your
    topic isn't special.
    """
    doc_topic = model_components['doc_topic']
    count=0
    for d in doc_topic:
        count += (np.argmax(d)==i)
    return count


######################################
# MULTI-TOPIC and FULL MODEL METRICS #
######################################

def jensen_shannon_divergence(topic_a,topic_b):
    _topic_a = topic_a/norm(topic_a,ord=1)
    _topic_b = topic_b/norm(topic_b,ord=1)
    _m = 0.5*(_topic_a+topic_b)
    return 0.5*(entropy(_topic_a,_m)+entropy(_topic_b,_m))

def cosine_distance(topic_a,topic_b):
    return cosine(topic_a,topic_b)
