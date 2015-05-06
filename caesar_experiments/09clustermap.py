"""
We plot the different distributions in scores for same-author and different-author pairs
in the dev set, and calculate whether they are statistically significantly different,
using the entire vocabulary.
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import seaborn as sb
matplotlib.rcParams['lines.linewidth'] = 0.8
from matplotlib.colors import rgb2hex
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette

sb.set_palette('Set1', 10, 0.80)
palette = sb.color_palette()
set_link_color_palette(map(rgb2hex, palette))

from verification.verification import Verification
from verification.evaluation import evaluate, evaluate_with_threshold, average_precision_score
from verification.evaluation import rank_predict
from verification.plotting import draw_tree
from verification.preprocessing import prepare_corpus, Dataset
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd

# select a data set
dev = "../data/caesar_dev"
test = "../data/caesar_test"

# we prepare the corpus
logging.info("preparing corpus")
X_dev = prepare_corpus(test)
X_test = prepare_corpus(test)


dm = 'minmax'
vsm = 'tf'

print dm
print vsm

verifier = Verification(random_state=1000,
                        metric=dm,
                        n_features=10000,
                        n_dev_pairs=0,
                        n_test_pairs=99999999,
                        vector_space_model=vsm,
                        balanced_pairs=False,
                        control_pairs=False)

logging.info("Starting verification [train / test]")
verifier.vectorize(X_dev, X_test)
train_results, test_results = verifier.predict(filter_imposters=False)
logging.info("Computing results")

test_df = verifier.get_distance_table(verifier.test_dists, verifier.test_pairs, "test")
test_df.to_csv("../outputs/caesar_test.csv")

test_df = pd.read_csv("../outputs/caesar_test.csv")
test_df = test_df.set_index("id")
test_df = test_df.applymap(lambda x:int(x*1000)).corr()

# heatmap plotting:
sb.heatmap(test_df)
ax = sb.plt.gca()
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(3)
sb.plt.savefig("../outputs/caesar_imposter_heatmap.pdf")
sb.plt.clf()

# clustermap plotting:
g = sb.clustermap(test_df)
ax = g.ax_heatmap
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(3)
g.savefig("../outputs/caesar_imposter_clustermap.pdf")
