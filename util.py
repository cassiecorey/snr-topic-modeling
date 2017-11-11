import pickle,os,metrics
from time import time

DATA_PATH = 'picklejar'

if not os.path.exists(DATA_PATH):
		os.makedirs(DATA_PATH)

def load_pickle(name):
	"""
	Loads a pickled object from the picklejar.

	Parameters:
	name	the name of the pickle object to load

	Returns the unpickled object or None if no
	pickle was found.
	"""
	try:
		with open(os.path.join(DATA_PATH,name),'rb') as f:
			return pickle.load(f)
	except:
		return None

def dump_pickle(obj,name):
	"""
	Save an object into the picklejar.

	Parameters:
	obj		the object to be stored as a pickle
	name 	the name for the pickle file
			this will be the name used to load later on

	Return True if successful, False otherwise.
	"""
	with open(os.path.join(DATA_PATH,name),'wb') as f:
		try:
			pickle.dump(obj,f)
			return True
		except:
			print("Problem dumping pickle.")
			return False

def alpha_separate(u,v,alpha):
	"""
	Separates two vectors by a parameter, alpha.
	The first vector stays the same, the second changes.
	In other words, u is the foreground/signal topic,
	and v is the background/noise topic.

	Parameters:
	alpha 	the distance to separate them by
	u,v 	vectors (must have same shape)
	"""
	u = np.array(u)
	v = np.array(v)
	assert(u.shape==v.shape)
	
	pass

def calc_metrics(model_components,verbose=True):
	"""
	Calculates the metrics for the given model.

	Returns:
	mets 	a dictionary of the topic metrics
	"""
	Kr = range(len(model_components['topic_word']))
	if verbose:
		print("Calculating rank1...",end='')
	t = time()
	r1 = [metrics.rank1(model_components,i) for i in Kr]
	if verbose:
		print("done in {:0.3f}s".format(time()-t))
		print("Calculating average word length...",end='')
	t = time()
	awl = [metrics.average_word_length(model_components,i) for i in Kr]
	if verbose:
		print("done in {:0.3f}s".format(time()-t))
		print("Calculating effective size...",end='')
	es = [metrics.effective_size(model_components,i) for i in Kr]
	if verbose:
		print("done in {:0.3f}s".format(time()-t))
		print("Calculating exclusivity...",end='')
	ex = [metrics.exclusivity(model_components,i) for i in Kr]
	if verbose:
		print("done in {:0.3f}s".format(time()-t))
		print("Calculating distance from uniform...",end='')
	du = [metrics.distance_from_uniform(model_components,i) for i in Kr]
	if verbose:
		print("done in {:0.3f}s".format(time()-t))
		print("Calculating distance from corpus...",end='')
	dc = [metrics.distance_from_corpus(model_components,i) for i in Kr]
	if verbose:
		print("done in {:0.3f}s".format(time()-t))
	return {"rank1":r1,
			"exclusivity":ex,
			"average_word_length":awl,
			"effective_size":es,
			"distance_from_uniform":du,
			"distance_from_corpus":dc}