import metrics

from bokeh.io import output_notebook, show
from bokeh.plotting import figure

from bokeh.models.widgets import Panel, Tabs
from bokeh.models import HoverTool, Legend, ColumnDataSource

K = 100

properties = ['num_docs','avg_doc_len','vocab_size',
              'readability','lexical_diversity',
              'stopword_presence']

topic_metrics = ['average_word_length','exclusivity',
                 'rank1','distance_from_uniform',
                 'distance_from_corpus','effective_size',
                 'top_words']

names = ['wine','brown','abc_rural','abc_science','genesis','inaugural','state_union']

def get_metric(components,met):
  """
  Get a metric by name from model components.
  """
  results = []
  metric_func = getattr(metrics,met)
  for i in range(K):
    results.append(metric_func(components,i))
  return results

def get_property(pcr,prop):
  """
  Get a property by name from a
  PropertiesCorpusReader object.
  """
  return getattr(pcr,prop)

def calculate_plotting_data(names,models,corpusreaders,verbose=True):
  """
  Calculates the necessary data for plotting.

  names is a list of keys for the following two dictionaries

  models is a dictionary containing all the models

  corpusreaders is a dictionary containing all the corpus readers
  """
  plotting_data = {}
  for n in names:
    if verbose:
      print(n)
      print("\tCalculating corpus properties...")
    for p in properties:
      val = get_property(corpusreaders[n],p)
      x = [val]*K
      plotting_data['{}_{}'.format(n,p)] = x
    if verbose:
      print("\tCalculating topic metrics...")
    for tm in topic_metrics:
      y = get_metric(models[n],tm)
      plotting_data['{}_{}'.format(n,tm)] = y
  return plotting_data

def plot_metric(metric_name, plotting_data, legend=True):
  """
  Plots a metric.
  """
  colors = {'wine':'red',
            'brown':'orange',
            'abc_rural':'yellow',
            'abc_science':'green',
            'genesis':'blue',
            'inaugural':'purple',
            'state_union':'pink'}

  
  
  property_tabs= []
  figs = {}
  for p in properties:
    fig = figure(x_axis_label=p,
                 y_axis_label=metric_name,
                 height=600,
                 width=800,
                 toolbar_location='above')
    
    # Make the labels legible when plots are downloaded
    fig.xaxis.axis_label_text_font_size = "30pt"
    fig.yaxis.axis_label_text_font_size = "30pt"
    fig.xaxis.major_label_text_font_size = "15pt"
    fig.yaxis.major_label_text_font_size = "15pt"
    fig.add_tools(hover)
    legend_items = []
    
    # One scatter plot for each corpus
    for n in names:
      circle = fig.circle(x='x',
                          y='y',
                          source=ColumnDataSource({'x':plotting_data['{}_{}'.format(n,p)],
                                                   'y':plotting_data['{}_{}'.format(n,metric_name)],
                                                   'top_three':[' '.join(w[:3]) for w in plotting_data['{}_top_words'.format(n)]]}),
                          size=20, color=colors[n], alpha=0.2)
      legend_items.append((n,[circle]))
    if legend:                        
      # Build and add the interactive plot legend, to the right of the plot
      legend = Legend(items=legend_items,
                      click_policy='hide',
                      label_text_font_size='20pt',
                      location=(10,-30),
                      glyph_height=50,
                      glyph_width=50,
                      border_line_alpha=0)
      fig.add_layout(legend,'right')
    figs[p] = fig
    property_tabs.append(Panel(child=fig, title=p))

  tabs = Tabs(tabs=property_tabs)

  show(tabs,notebook_handle=True)