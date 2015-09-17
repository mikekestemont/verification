from __future__ import print_function

import random
import sys
from operator import itemgetter
from collections import Counter
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sb
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from verification.verification import Verification
from verification.evaluation import evaluate
from verification.preprocessing import prepare_corpus, split_corpus, Dataset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering

from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, show, output_file, save
from bokeh.charts import Bar
from bokeh.io import output_file, show, vplot, save
from bokeh.plotting import figure
from bokeh.models import Axis


data_path = '../data/'
corpus = 'soldier_letters'

logging.info('preparing corpus')
verif_dataset = prepare_corpus(data_path+corpus)

fit = 0
if fit:
    """
    We fit a vectorizer with the best parametrization
    we obtained during the development phase.
    """
    verifier = Verification(random_state=1066,
                            metric='minmax',
                            feature_type='chars',
                            ngram_range=4,
                            sample_authors=False,
                            sample_features=False,
                            n_features=10000,
                            n_dev_pairs=1000000000000,
                            vector_space_model='std',
                            balanced_pairs=False)
    verifier.vectorize(verif_dataset)
    dev_results = verifier.fit()
    dev_f, dev_p, dev_r, dev_t = evaluate(dev_results)
    max_f = np.nanmax(dev_f)
    print('\t\t + F1 (pairwise):', max_f)

    print('getting distance table')
    df = verifier.get_distance_table(verifier.dev_dists, verifier.dev_pairs, 'dev')
    df.to_csv('outputs/dm_no_sampl.csv')
    print('saved dist table!')

df = pd.read_csv('outputs/dm_no_sampl.csv')
df = df.set_index('id')
print('loaded dist table!')

top_scores = 0
if top_scores:
    scores = []
    for author1 in df.index:
        for author2 in df.columns:
            if author1.split('_', 1)[0] != author2.split('_', 1)[0]:
                sc = df[author1][author2]
                if sc > 0:
                    scores.append((author1, author2, sc))
    scores = sorted(scores, key=itemgetter(2))
    for sc in scores[:250]:
        print(sc)

draw_tsne = 0
if draw_tsne:
    print("plotting tsne!")
    author_labels = [l.split('_', 1)[0] for l in list(df.columns)]
    full_info = list(df.columns)
    X = df.as_matrix()
    tsne = TSNE(n_components=2, random_state=1987, verbose=1, n_iter=2500, perplexity=5.0,
                early_exaggeration=2.0, learning_rate=100, metric='precomputed')
    tsne_projection = tsne.fit_transform(X)

    # clustering on top (for colouring):
    clusters = AgglomerativeClustering(n_clusters=8).fit_predict(tsne_projection)
    # get color palette:
    colors = sb.color_palette('husl', n_colors=8)
    colors = [tuple([c * 256 for c in color]) for color in colors]
    colors = ['#%02x%02x%02x' % colors[i] for i in clusters]
    TOOLS="pan,wheel_zoom,reset,hover,box_select,save"
    source = ColumnDataSource(data=dict(x=tsne_projection[:,0], y=tsne_projection[:,1], name=full_info))
    output_file("outputs/embeddings_no_sampl.html")
    p = figure(title="Auteurskaart", tools=TOOLS,
               plot_width=1000, title_text_font="Arial", 
               plot_height=800, outline_line_color="white")
    p.circle(x=tsne_projection[:,0], y=tsne_projection[:,1], source=source, size=8, color=colors,
               fill_alpha=0.9, line_color=None)

    counter = Counter(author_labels)
    for name, x, y in zip(author_labels, tsne_projection[:,0], tsne_projection[:,1]):
        if counter[name] > 1:
            p.text(x, y, text=[name], text_align="center", text_font_size="10pt")


    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [("info", "@name")]
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_label_text_font_size = '0pt'
    # Turn off tick marks
    p.axis.major_tick_line_color = None
    p.axis[0].ticker.num_minor_ticks = 0
    p.axis[1].ticker.num_minor_ticks = 0
    save(p)

draw_clustermap = 1
if draw_clustermap:
    print('plotting clustermap')
    df = df.applymap(lambda x:int(x*1000)).corr()
    g = sb.clustermap(df)
    ax = g.ax_heatmap
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(1)
    g.savefig("outputs/clustermap.svg")

