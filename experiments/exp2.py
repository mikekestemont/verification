"""
We plot the different distributions in scores for same-author and different-author pairs
in the dev set, and calculate whether they are statistically significantly different,
using the entire vocabulary.
"""

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

from scipy.stats import ks_2samp

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import seaborn as sb

from verification.verification import Verification
from verification.smooth import *
from verification.evaluation import evaluate, evaluate_with_threshold, average_precision_score
from verification.evaluation import rank_predict
from verification.plotting import plot_test_results
from verification.preprocessing import prepare_corpus, Dataset
from sklearn.cross_validation import train_test_split
import numpy as np

# select a data set
train = "../data/du_essays"
test = train
print "Using data under: "+train

# we prepare the corpus
logging.info("preparing corpus")
if train == test:
    data = prepare_corpus(train)
    train_texts, test_texts, train_titles, test_titles, train_authors, test_authors = train_test_split(
        data.texts, data.titles, data.authors, test_size=0.5, random_state=1000)
    X_train = Dataset(train_texts, train_titles, train_authors)
    X_test = Dataset(test_texts, test_titles, test_authors)
    print len(X_train.texts)
else:
    X_train = prepare_corpus(train)
    X_test = prepare_corpus(test)

# we determine the size of the entire vocabulary
V = int(len(set(sum(X_train.texts, []) + sum(X_test.texts, [])))/2)

vsms = ('std', 'plm', 'tf', 'idf')
dms  = ('euclidean', 'cityblock', 'minmax')

# set fig params
fig = sb.plt.figure(figsize=(len(dms), len(vsms)))
cnt = 0
outer_grid = gridspec.GridSpec(len(dms), len(vsms), wspace=0.1, hspace=0.1)
c1, c2 = sb.color_palette("Set1")[:2]

for dm_cnt, distance_metric in enumerate(dms):
    for vsm_cnt, vector_space_model in enumerate(vsms):
        verifier = Verification(random_state=1000,
                                sample_features=False,
                                metric=distance_metric,
                                sample_authors=False,
                                n_features=V,
                                n_train_pairs=500,
                                n_test_pairs=500,
                                em_iterations=100,
                                vector_space_model=vector_space_model,
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
        train_f, train_p, train_r, train_t = evaluate(train_results)
        smooth_train_f = smooth(train_f, window_len=25, window='flat')
        best_t = train_t[np.nanargmax(smooth_train_f)]

        test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t)
        print "\t* "+vector_space_model+" & "+distance_metric+":"
        print "\t\t- F-score: "+str(test_f)
        print "\t\t- Precision: "+str(test_p)
        print "\t\t- Recall: "+str(test_r)

        same_author_densities = np.asarray([sc for c, sc in train_results if c == "same_author"])
        diff_author_densities = np.asarray([sc for c, sc in train_results if c == "diff_author"])
        D, p = ks_2samp(same_author_densities, diff_author_densities)
        print "\t\t- KS: D = "+str(D)+" (p = "+str(p)+")"
        sb.set_style("dark")
        ax = sb.plt.Subplot(fig, outer_grid[cnt])
        ax.set_xlim([0, 1])
        sb.kdeplot(diff_author_densities, shade=True, legend=False, c=c1, ax=ax, lw=0.5)
        sb.kdeplot(same_author_densities, shade=True, legend=False, c=c2, ax=ax, lw=0.5)
        if vsm_cnt == 0:
            ax.set_ylabel(distance_metric, fontsize=5)
        if dm_cnt == 0:
            ax.set_title(vector_space_model, fontsize=5)
        ax.xaxis.set_major_formatter(sb.plt.NullFormatter())
        ax.yaxis.set_major_formatter(sb.plt.NullFormatter())
        from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
        at = AnchoredText("F1: "+str(round(test_f, 3))+"\nKS: "+str(round(D, 3)), prop=dict(size=3), frameon=True, loc=2)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        sb.axes_style()
        fig.add_subplot(ax)
        cnt+=1
sb.plt.savefig("../plots/exp2_distributions.pdf")
