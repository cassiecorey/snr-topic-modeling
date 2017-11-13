# UTILITIES
import pickle,os,metrics
import fixed_model as fm
import numpy as np
import pandas as pd
from time import time

# PLOTTING
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import (HoverTool,
                          Legend,
                          ColumnDataSource,
                          LinearColorMapper,
                          BasicTicker,
                          PrintfTickFormatter,
                          ColorBar)

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
	signal = alpha*noise + (1-alpha)*signal
	The first vector stays the same, the second changes.
	In other words, u is the foreground/signal topic,
	and v is the background/noise topic.

	Parameters:
	u 			the signal vector
	v 			the noise vector
	alpha 		the distance to separate them by
	"""
	return alpha*v + (1-alpha)*u

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

def generate(model_components,N):
	"""
	Generates documents of length N from the model components.
	Number of documents is determined by length of doc_topic component.
	"""
	topic_word = model_components['topic_word']
	features = model_components['features']
	doc_topic = model_components['doc_topic']

	# Normalize topic_word distribution
	row_sums = topic_word.sum(axis=1)
	norm_topic_word = topic_word/row_sums[:,np.newaxis]

	topics = range(len(topic_word))
	data_samples = []
	for d in doc_topic:
		new_doc = ""
		# Select a topic from topic distribution
		topic = np.random.choice(topics,p=d)
		for n in range(N):
			# Select a word from the word distribution
			word = np.random.choice(features,p=norm_topic_word[topic])
			new_doc += word + " "
		data_samples.append(new_doc)
	return data_samples

def build_components(topic_word,features,D,N):
	"""
	Builds model components directly from topic word distribution.
	"""
	K = topic_word.shape[0] # number of topics
	V = topic_word.shape[1] # vocab size
	row_sums = topic_word.sum(axis=1)
	norm_topic_word = topic_word/row_sums[:,np.newaxis]

	# CURRENTLY EQUAL DISTRIBUTION OVER TOPICS
	doc_topic = np.array([[1/K for k in range(K)] for d in range(D)])
	doc_word = np.array([[1/K for w in range(V)] for d in range(D)])
	for d in range(D):
		topic = np.random.choice(range(K),p=doc_topic[d])
		for n in range(N):
			topic_word = norm_topic_word[topic]
			feat_idx = np.random.choice(range(V),p=norm_topic_word[topic])
			doc_word[d][feat_idx] += 1

	model_components = {}
	model_components['topic_word'] = norm_topic_word
	model_components['features'] = features
	model_components['doc_word'] = doc_word
	model_components['doc_topic'] = doc_topic
	return model_components

def build_metrics_tabs(model_components):
	topic_metrics = calc_metrics(model_components,verbose=False)
	source_dict = topic_metrics.copy()
	K = model_components['topic_word'].shape[0]
	source_dict['x'] = range(K)
	source = ColumnDataSource(source_dict)

	top_words = fm.get_top_words(model_components)
	source.data['top_three'] = [' '.join(top_words[t][:3]) for t in range(K)]
	hover = HoverTool(tooltips=[('top words','@top_three')])

	metric_tabs = []
	for m in topic_metrics.keys():
	    fig = figure(x_axis_label='Topic Number',
	                 y_axis_label=m,
	                 height=600,
	                 width=600,
	                 toolbar_location='above')
	    fig.add_tools(hover)
	    fig.vbar(x='x',top=m,width=0.75,source=source)
	    metric_tabs.append(Panel(child=fig, title=m))

	distance_algorithms = ["jensen_shannon_divergence",
                       	   "kullback_leibler_divergence",
                      	   "cosine_distance"]

	for dist_alg in distance_algorithms:
	    df = pd.DataFrame(columns=list(range(K)),index=list(range(K)))
	    topic_word = model_components['topic_word']
	    for i in range(K):
	        for j in range(K):
	            func = getattr(metrics,dist_alg)
	            df.at[i,j] = func(topic_word[i],topic_word[j])
	    df.index.name="TopicA"
	    df.columns.name="TopicB"
	    topicsA = list(df.index)
	    topicsB = list(df.columns)
	    p_df = pd.DataFrame(df.stack(),columns=['dist']).reset_index()
	    p_df['x'] = p_df["TopicA"] + 0.5
	    p_df['y'] = p_df["TopicB"] + 0.5
	    TOOLS = "hover,save"
	    mapper = LinearColorMapper(palette='Spectral10',low=0,high=1)
	    color_bar = ColorBar(color_mapper=mapper,
	                         major_label_text_font_size='10pt',
	                         border_line_color=None,
	                         location=(0,0))
	    source = ColumnDataSource(p_df)
	    p = figure(title=dist_alg,
	               x_range=[str(i) for i in topicsA],y_range=[str(i) for i in topicsB],
	               tools=TOOLS, toolbar_location='above')
	    p.grid.grid_line_color=None
	    p.axis.axis_line_color=None
	    p.axis.major_tick_line_color=None
	    p.axis.major_label_text_font_size='10pt'
	    p.axis.major_label_standoff=0
	    p.rect(x="x",y="y",width=1,height=1,source=source,
	           fill_color={'field':'dist','transform':mapper},line_color=None)
	    p.add_layout(color_bar, 'right')
	    p.select_one(HoverTool).tooltips = [
	        ('Topics','@TopicA and @TopicB'),
	        (dist_alg,'@dist')
	    ]
	    metric_tabs.append(Panel(child=p,title=dist_alg))

	return metric_tabs