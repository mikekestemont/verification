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
"""
# select a data set
dev = "../data/caesar_dev"
test = "../data/caesar_test"

# we prepare the corpus
logging.info("preparing corpus")
X_dev = prepare_corpus(dev)
X_test = prepare_corpus(test)

#best_feats_df = pd.read_csv("../plots/caesar_nf.csv")
#best_feats_df = best_feats_df.set_index("distance metric")
#print(best_feats_df)

dm = 'cityblock'
vsm = 'tf'

print dm
print vsm
#print best_feats_df.loc[dm,vsm]
verifier = Verification(random_state=1000,
                        sample_features=True,
                        metric=dm,
                        sample_authors=True,
                        sample_iterations=100,
                        n_features=10000, # based on optimal training F1
                        n_dev_pairs=2,
                        n_test_pairs=100000,
                        em_iterations=100,
                        random_prop=0.5,
                        vector_space_model=vsm,
                        weight=0.2,
                        n_actual_imposters=10,
                        n_potential_imposters=60,
                        eps=0.01,
                        norm="l2",
                        top_rank=10,
                        balanced_pairs=False)
logging.info("Starting verification [train / test]")
verifier.fit(X_dev, X_test)
train_results, test_results = verifier.verify()
logging.info("Computing results")

test_df = verifier.get_distance_table(verifier.test_dists, verifier.test_pairs, "test")
test_df.to_csv("../plots/caesar_test.csv")
"""
test_df = pd.read_csv("../plots/caesar_test.csv")
test_df = test_df.set_index("id")
test_df = test_df.applymap(lambda x:int(x*1000)).corr()

# heatmap plotting:
sb.heatmap(test_df)
ax = sb.plt.gca()
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(3)
sb.plt.savefig("../plots/caesar_imposter_heatmap.pdf")
sb.plt.clf()

# clustermap plotting:
g = sb.clustermap(test_df)
ax = g.ax_heatmap
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(3)
g.savefig("../plots/caesar_imposter_clustermap.pdf")