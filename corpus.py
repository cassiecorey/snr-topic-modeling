from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

from scipy.spatial.distance import cosine
from collections import Counter

import readability,os
import numpy as np

# List of currently accessible properties.
PROPERTIES = ['num_docs','avg_doc_len','vocab_size',
              'readability','lexical_diversity',
              'stopword_presence']

def from_strings(name, doc_strings):
	"""
	Converts a list of strings into a PropertiesCorpusReader object.
	"""
	if not os.path.exists(name):
		os.makedirs(name)

	fileids = []
	for i in range(len(doc_strings)):
		fn = '{:04}.txt'.format(i)
		fileids.append(fn)
		with open(os.path.join(name,fn),'w') as f:
			f.write(doc_strings[i])
		f.close()
	plaintext_reader = PlaintextCorpusReader(name,fileids)
	properties_reader = PropertiesCorpusReader(plaintext_reader,verbose=False)
	return properties_reader

class PropertiesCorpusReader(PlaintextCorpusReader):
	"""
	This is a wrapper of the NLTK plaintext corpus reader
	which already provides a lot of useful functions.

	Similar to most of the NLTK readers, you can get
	characteristics for the full corpus or for a specific document.

	Attributes:
		number of documents
		average document length
		distance from uniform distribution
		vocabulary size
		readability (smog index)
		lexical_diversity
		stopword_presence
	"""

	def __init__(self, plaintext_reader, verbose=True, init_properties=True):
		"""
		Initialize a PropertiesCorpusReader object.
		It's faster if you set init_properties field to False.
		"""
		PlaintextCorpusReader.__init__(self,
									   plaintext_reader.root,
									   plaintext_reader.fileids(),
									   word_tokenizer=plaintext_reader._word_tokenizer,
									   sent_tokenizer=plaintext_reader._sent_tokenizer,
									   para_block_reader=plaintext_reader._para_block_reader,
									   encoding=plaintext_reader._encoding)

		self.num_docs = len(self.fileids())
		self.avg_doc_len = np.array([len(self.words(f)) for f in self.fileids()]).mean()
		self.vocab_size = len(set(self.words()))
		if init_properties:
			if verbose:
				print("Calculating properties...")
			self.readability = self.readability()
			if verbose:
				print("\tReadability calculated.")
			self.distance_from_uniform = self.distance_from_uniform()
			if verbose:
				print("\tDistance from uniform calculated.")
			self.lexical_diversity = self.lexical_diversity()
			if verbose:
				print("\tLexical diversity calculated.")
			self.stopword_presence = self.stopword_presence()
			if verbose:
				print("\tStopword presence calculated.")

	def readability(self,doc=None):
		measures = readability.getmeasures(self.raw(doc))
		return measures['readability grades']['SMOGIndex']

	def distance_from_uniform(self,doc=None,stopwords=False):
		"""
		Distance of the corpus from a uniform distribution over
		the vocabulary size.
		"""
		words = word_tokenize(self.raw(doc))
		wcounts = list(FreqDist(w.lower() for w in words).values())
		uniform = [1/len(wcounts) for i in range(len(wcounts))]
		return cosine(wcounts,uniform)	

	def lexical_diversity(self,doc=None):
		"""
		A measure of how many different words are used in a text.
		
		If a text uses different vocabulary for the same idea,
		it will show higher complexity and diversity.
		"""
		words = self.words(doc)
		return len(set(words))/len(words)

	def stopword_presence(self,doc=None):
		"""
		What is the frequency of stopwords in the corpus?
		
		Returns the ratio of stopwords to other words.
		"""
		stopword_count = 0
		counter = Counter(self.words(doc))
		u = set(self.words(doc))
		v = set(stopwords.words())
		for w in u.intersection(v):
			stopword_count += counter[w]
		return stopword_count/len(self.words(doc))

	def word_distribution(doc=None):
		"""
		What is the distribution of words in the corpus?

		Returns an alphabetized distribution of the words 
		"""
		words = self.words()
		pass


	def raw_docs(self):
		"""
		Returns a list of the raw documents in the corpus.
		"""
		return [self.raw(d) for d in self.fileids()]

	# This is just a nice function to have for printing pretty stats.
	def print_properties(self):
		print("Number of documents: {}".format(self.num_docs))
		print("Average document length: {:.4f}".format(self.avg_doc_len))
		print("Vocab size: {}".format(self.vocab_size))
		print("Readability: {:.4f}".format(self.readability))
		print("Distance from uniform: {:.4f}".format(self.distance_from_uniform))
		print("Lexical diversity: {:.4f}".format(self.lexical_diversity))
		print("Stopword presence: {:.4f}".format(self.stopword_presence))