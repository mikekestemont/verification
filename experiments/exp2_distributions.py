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

from verification.verification import Verification
from verification.smooth import *
from verification.evaluation import evaluate, evaluate_with_threshold, average_precision_score
from verification.evaluation import rank_predict
from verification.plotting import plot_test_results
from verification.preprocessing import prepare_corpus, split_corpus

from supersmoother import SuperSmoother
from scipy.stats import ks_2samp

random_state = 1000
dev = "../data/du_essays"
print "Using data under: "+dev

# we prepare the corpus
logging.info("preparing corpus")
X_dev, X_test = split_corpus(prepare_corpus(dev), controlled=True)

# we determine the upper half of the vocabulary
V = int(len(set(sum(X_dev.texts, []) + sum(X_test.texts, [])))/2.0)

vsms = ('std', 'plm', 'tf', 'tfidf', 'bin')
dms  = ('euclidean', 'cityblock', 'minmax')

# set fig params
fig = sb.plt.figure(figsize=(len(dms), len(vsms)))
cnt = 0
outer_grid = gridspec.GridSpec(len(dms), len(vsms), wspace=0.1, hspace=0.1)
c1, c2 = sb.color_palette("Set1")[:2]

for dm_cnt, distance_metric in enumerate(dms):
    for vsm_cnt, vector_space_model in enumerate(vsms):
        verifier = Verification(random_state=random_state,
                                sample_features=False,
                                metric=distance_metric,
                                sample_authors=False,
                                n_features=V,
                                n_dev_pairs=250,
                                n_test_pairs=250,
                                em_iterations=100,
                                vector_space_model=vector_space_model,
                                weight=0.2,
                                n_actual_imposters=10,
                                eps=0.01,
                                norm="l2",
                                top_rank=1,
                                balanced_pairs=True)
        logging.info("Starting verification [dev / test]")
        verifier.fit(X_dev, X_test)
        dev_results, test_results = verifier.verify()

        logging.info("Computing results")
        dev_f, dev_p, dev_r, dev_t = evaluate(dev_results)
        
        model = SuperSmoother()
        model.fit(dev_t, dev_f)
        smooth_dev_f = model.predict(dev_t)

        best_t = dev_t[np.nanargmax(smooth_dev_f)]

        test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t)
        print "\t* "+vector_space_model+" & "+distance_metric+":"
        print "\t\t- F-score: "+str(test_f)
        print "\t\t- Precision: "+str(test_p)
        print "\t\t- Recall: "+str(test_r)
        # distribution of unsmoothed scores:
        same_author_densities = np.asarray([sc for c, sc in dev_results if c == "same_author"])
        diff_author_densities = np.asarray([sc for c, sc in dev_results if c == "diff_author"])
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
