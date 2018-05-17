import os,sys,corpus
import numpy as np

from corpus import PropertiesCorpusReader
from nltk.corpus import stopwords,webtext,brown
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

from metrics_model import MetricsModel
from collections import Counter

from time import time

N_TOPICS = 15

stoplist = stopwords.words('english')

metrics_list = ['top_words','exclusivity','rank1','dist_uni','eff_size']

local_corpora = ['abc_science','abc_rural','wine','brown']
brown_corpora = ['webtext']

CORPUS_NAME = sys.argv[1]

def stopwords_count(tokens):
	stopword_count = 0
	counter = Counter(tokens)
	u = set(tokens)
	for w in u.intersection(stoplist):
		stopword_count += counter[w]
	return stopword_count

def stopwords_presence(tokens):
	return stopwords_count(tokens)/len(tokens)

"""
STEP 1: Build a base model.
"""
mm = MetricsModel()
print("Loaded Model")
"""
STEP 2: Load a corpus.
"""
print("EXPERIMENT BEING RUN ON: ",CORPUS_NAME)
if CORPUS_NAME in local_corpora:
	CORPUS_DIR = os.path.join('corpus',CORPUS_NAME)
	fileids = []
	for f in os.listdir(CORPUS_DIR):
		if os.path.isfile(os.path.join(CORPUS_DIR,f)):
			fileids.append(f)
	cr = PlaintextCorpusReader(CORPUS_DIR,fileids)
	pcr = PropertiesCorpusReader(cr)
elif CORPUS_NAME in brown_corpora:
	CORPUS_DIR = getattr(eval(CORPUS_NAME),'root')
	cr = eval(CORPUS_NAME)
	pcr = PropertiesCorpusReader(cr)

print()
print("Loaded Corpus")
print()

with open("{}_ind_experiment_0.csv".format(CORPUS_NAME),'w') as f1, open("{}_pair_experiment_0.csv".format(CORPUS_NAME),'w') as f2:
	# Column headers
	f1.write("num_docs,doc_len,sw_pres,topic,top_words,dist_uni,eff_size,exclusivity,rank1")
	f1.write("\n")
	f2.write("num_docs,doc_len,sw_pres,topic_pair,cos,kld,jsd")
	f2.write("\n")

	"""
	STEP 3: Modify the corpus.
	"""
	# 3.a. Modify the number of documents
	d_step = int(pcr.num_docs/10)
	d_range = np.arange(d_step,pcr.num_docs,d_step)
	t00 = time()
	t0 = time()
	for d in d_range:
		d_fileids = np.random.choice(pcr.fileids(),d)
		d_cr = PlaintextCorpusReader(CORPUS_DIR,d_fileids)
		d_pcr = PropertiesCorpusReader(d_cr,verbose=False)
		print("Finished 3.a in {:0.3f}s".format(time()-t0))
		# 3.b. Modify the length of documents
		l_step = int(pcr.avg_doc_len/10)
		l_range = np.arange(l_step,pcr.avg_doc_len,l_step)
		t0 = time()
		for l in l_range:
			l_strings = []
			for fn in d_pcr.fileids():
				words = d_pcr.words(fn)
				l_strings.append(' '.join(np.random.choice(words,int(l))))
			l_corpus_dir = os.path.join("{}_{}".format(CORPUS_DIR,d),'doc_len_'.format(int(l)))
			l_pcr = corpus.from_strings(l_corpus_dir,l_strings)
			print("Finished 3.b in {:0.3f}s".format(time()-t0))
			# 3.c. Modify the presence of stopwords
			s_range = np.arange(0,1,0.1)
			t0 = time()
			for s in s_range:
				s_strings = []
				for fn in l_pcr.fileids():
					tokens = [w.lower() for w in list(l_pcr.words(fn))]
					n_s = len(words)*s # Proportion of stopwords we want
					s_p = stopwords_presence(tokens)
					if s_p < s: # Add stopwords
						to_add = int((s_p-n_s)/(s-1))
						tokens.extend(np.random.choice(stoplist,to_add))
					elif s_p > s: # Remove stopwords (one at a time)
						while stopwords_presence(tokens) > s_p:
							sw = np.random.choice(set(tokens).intersection(stoplist))
							tokens.remove(sw)
					s_strings.append(' '.join(tokens))
				s_corpus_dir = os.path.join("{}_{}_{}".format(CORPUS_DIR,d,l),'sw_pres_{:0.1f}'.format(s))
				s_pcr = corpus.from_strings(s_corpus_dir,s_strings)
				print("Finished 3.c in {:0.3f}s".format(time()-t0))
				"""
				STEP 4: Fit model to modified corpus.
				"""
				t0 = time()
				mm.fit_from_samples(s_pcr.raw_docs(),N_TOPICS)
				print("Fit from samples in {:0.3f}s".format(time()-t0))
				"""
				STEP 5: Record resulting topic metrics.
				"""
				print("Writing topic metrics...",end='')
				t0 = time()
				for i in range(N_TOPICS):
					# Write individual topic metrics
					f1.write("{},{},{},{},".format(d,l,s,i)) # Independent variables (minus topic num)
					f1.write("{},".format('_'.join(mm.top_words(i,n=3)))) # Top 3 words
					f1.write("{},".format(mm.distance_from_uniform(i)))
					f1.write("{},".format(mm.effective_size(i)))
					f1.write("{},".format(mm.exclusivity(i)))
					f1.write("{},".format(mm.rank1(i)))
					f1.write("\n") # End line.
					for j in range(N_TOPICS):
						# Write pair topic metrics
						f2.write("{},{},{},{}_{},".format(d,l,s,i,j))
						f2.write("{},".format(mm.cosine_distance(i,j)))
						f2.write("{},".format(mm.kullback_leibler_divergence(i,j)))
						f2.write("{}".format(mm.jensen_shannon_divergence(i,j)))
						f2.write("\n")
				print("done in {:0.3f}s".format(time()-t0))
				print()
		print("Done with: {} in {:0.3f}s".format(d,time()-t00))
		print()
		print()
