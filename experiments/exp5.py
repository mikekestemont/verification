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
train = "../data/caesar_background"
test = "../data/caesar_devel"

# we prepare the corpus
logging.info("preparing corpus")
X_train = prepare_corpus(train)
X_test = prepare_corpus(test)

dms = ('minmax', 'euclidean', 'cityblock')
vsms = ('std', 'plm', 'tf', 'idf')

best_feats_df = pd.read_csv("../plots/caesar_nf.csv")
best_feats_df = best_feats_df.set_index("distance metric")
print(best_feats_df)

fig = plt.figure()
cnt = 0
outer_grid = gridspec.GridSpec(len(dms), len(vsms))


for dm_cnt, dm in enumerate(dms):
    print dm
    for vsm_cnt, vsm in enumerate(vsms):
        print vsm
        print best_feats_df.loc[dm,vsm]
        verifier = Verification(random_state=1000,
                                sample_features=False,
                                metric=dm,
                                sample_authors=False,
                                n_features=best_feats_df.loc[dm,vsm], # based on optimal training F1
                                n_train_pairs=1,
                                n_test_pairs=1000,
                                em_iterations=100,
                                vector_space_model=vsm,
                                weight=0.2,
                                n_actual_imposters=10,
                                eps=0.01,
                                norm="l2",
                                top_rank=1,
                                balanced_pairs=True)
        logging.info("Starting verification [train / test]")
        verifier.fit(X_train, X_test)
        train_results, test_results = verifier.verify()
        logging.info("Computing results")
        
        test_df = verifier.get_distance_table(verifier.test_dists, verifier.test_pairs, "test")
        ax = plt.Subplot(fig, outer_grid[cnt])
        linkage_matrix = linkage(test_df, 'ward')
        f = dendrogram(linkage_matrix,
                   truncate_mode='lastp',
                   show_leaf_counts=True,
                   ax=ax,
                   orientation='right',
                   labels=test_df.columns,
                   leaf_font_size=0.5,
                   link_color_func=None,
                   color_threshold=np.inf)
        tickL = ax.yaxis.get_ticklabels()
        for t in tickL:
            t.set_fontsize(7)
            t.set_color('grey')
        ax.get_xaxis().set_ticks([])
        if vsm_cnt == 0:
            ax.set_ylabel(dm, fontsize=8)
        if dm_cnt == 0:
            ax.set_title(vsm, fontsize=8)
        ax.set_axis_bgcolor('white')
        no_spine = {'left': True, 'bottom': True, 'right': False, 'top': True}
        sb.despine(**no_spine)
        fig.add_subplot(ax)
        cnt+=1
plt.tight_layout()
plt.savefig("../plots/caesar_tree.pdf")
            
