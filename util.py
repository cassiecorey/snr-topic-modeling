import pickle,os

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

