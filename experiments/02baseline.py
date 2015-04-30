"""Baseline experiment on modern corpora. Used to generate Figs. 1.x and Tables 2.x """

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.ERROR)

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
from verification.preprocessing import prepare_corpus, split_corpus

random_state = 1000
data_path = "../data/"
corpus = "gr_articles" #"du_essays"
n_dev_pairs = 500
n_features = 10000

logging.info("preparing corpus")
X_dev, X_test = split_corpus(prepare_corpus(data_path+corpus), controlled="authors", random_state=random_state)

print corpus
vsms = ('std', 'tf', 'tfidf', 'bin', 'plm')
dms = ('minmax', 'euclidean', 'cityblock')

# set fig params
fig = sb.plt.figure(figsize=(len(vsms), len(dms)))
cnt = 0
outer_grid = gridspec.GridSpec(len(vsms), len(dms), wspace=0.1, hspace=0.1)
c1, c2 = sb.color_palette("Set1")[:2]

# first baseline:
df = pd.DataFrame(columns=["vector space model"]+list(dms))

for vsm_cnt, vsm in enumerate(vsms):
    print("\t+ "+vsm)
    fscore_row = [vsm]
    for dm_cnt, dm in enumerate(dms):
        print("* "+dm)
        verifier = Verification(random_state=random_state,
                                metric=dm,
                                feature_type="words",
                                sample_authors=False,
                                sample_features=False,
                                n_features=n_features,
                                n_test_pairs=0,
                                n_dev_pairs=n_dev_pairs,
                                em_iterations=100,
                                vector_space_model=vsm,
                                weight=0.2,
                                eps=0.01,
                                norm="l2",
                                balanced_pairs=True)
        logging.info("Starting verification [dev / test]")
        verifier.vectorize(X_dev, X_test)
        dev_results = verifier.fit()

        logging.info("Computing results")
        dev_f, dev_p, dev_r, dev_t = evaluate(dev_results)
        max_f = np.nanmax(dev_f)
        print "\t\t + F1: "+str(max_f)
        fscore_row.append(format(max_f*100, '.1f'))

        # distribution of scores:
        same_author_densities = np.asarray([sc for c, sc in dev_results if c == "same_author"])
        diff_author_densities = np.asarray([sc for c, sc in dev_results if c == "diff_author"])

        D, p = ks_2samp(same_author_densities, diff_author_densities)
        print "\t\t- KS: D = "+str(D)+" (p = "+str(p)+")"
        sb.set_style("dark")
        ax = sb.plt.Subplot(fig, outer_grid[cnt])
        ax.set_xlim([0, 1])
        sb.kdeplot(diff_author_densities, shade=True, legend=False, c=c1, ax=ax, lw=0.5)
        sb.kdeplot(same_author_densities, shade=True, legend=False, c=c2, ax=ax, lw=0.5)
        if dm_cnt == 0:
            ax.set_ylabel(vsm, fontsize=5)
        if vsm_cnt == 0:
            ax.set_title(dm, fontsize=5)
        ax.xaxis.set_major_formatter(sb.plt.NullFormatter())
        ax.yaxis.set_major_formatter(sb.plt.NullFormatter())
        from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
        at = AnchoredText("F1: "+str(format(max_f*100, '.1f'))+"\nKS: "+str(format(D, '.3f')), prop=dict(size=3), frameon=True, loc=2)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        sb.axes_style()
        fig.add_subplot(ax)
        cnt+=1
    df.loc[vsm_cnt] = fscore_row

# save
sb.plt.savefig("../outputs/"+corpus+"_distribs_baseline.pdf")

# take care of the tables:
df = df.set_index("vector space model")
# row and col names:
df.columns.name = "distance metric"
df.index.name = "distance metric"
df.to_csv("../outputs/"+corpus+"_baseline.csv")
print("=== baseline f1-scores ===")
print str(df.to_latex())

#####################################################################
### now imposters ###################################################
#####################################################################
# clear plot first:
sb.plt.clf()

# set fig params
fig = sb.plt.figure(figsize=(len(vsms), len(dms)))
cnt = 0
outer_grid = gridspec.GridSpec(len(vsms), len(dms), wspace=0.1, hspace=0.1)
c1, c2 = sb.color_palette("Set1")[:2]

df = pd.DataFrame(columns=["vector space model"]+list(dms))

for vsm_cnt, vsm in enumerate(vsms):
    print("\t+ "+vsm)
    fscore_row = [vsm]
    for dm_cnt, dm in enumerate(dms):
        print("* "+dm)
        verifier = Verification(random_state=random_state,
                                metric=dm,
                                feature_type="words",
                                sample_authors=True,
                                sample_iterations=100,
                                sample_features=True,
                                n_features=n_features,
                                random_prop=0.5,
                                n_test_pairs=0,
                                n_dev_pairs=n_dev_pairs,
                                em_iterations=100,
                                vector_space_model=vsm,
                                n_potential_imposters=120,
                                n_actual_imposters=10,
                                weight=0.2,
                                top_rank=10,
                                eps=0.01,
                                norm="l2",
                                balanced_pairs=True)
        logging.info("Starting verification [dev / test]")
        verifier.vectorize(X_dev, X_test)
        dev_results = verifier.fit()

        logging.info("Computing results")
        dev_f, dev_p, dev_r, dev_t = evaluate(dev_results)
        max_f = np.nanmax(dev_f)
        print "\t\t + F1: "+str(max_f)
        fscore_row.append(format(max_f*100, '.1f'))

        # distribution of scores:
        same_author_densities = np.asarray([sc for c, sc in dev_results if c == "same_author"])
        diff_author_densities = np.asarray([sc for c, sc in dev_results if c == "diff_author"])

        D, p = ks_2samp(same_author_densities, diff_author_densities)
        print "\t\t- KS: D = "+str(D)+" (p = "+str(p)+")"
        sb.set_style("dark")
        ax = sb.plt.Subplot(fig, outer_grid[cnt])
        ax.set_xlim([0, 1])
        sb.kdeplot(diff_author_densities, shade=True, legend=False, c=c1, ax=ax, lw=0.5)
        sb.kdeplot(same_author_densities, shade=True, legend=False, c=c2, ax=ax, lw=0.5)
        if dm_cnt == 0:
            ax.set_ylabel(vsm, fontsize=5)
        if vsm_cnt == 0:
            ax.set_title(dm, fontsize=5)
        ax.xaxis.set_major_formatter(sb.plt.NullFormatter())
        ax.yaxis.set_major_formatter(sb.plt.NullFormatter())
        from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
        at = AnchoredText("F1: "+str(format(max_f*100, '.1f'))+"\nKS: "+str(format(D, '.3f')), prop=dict(size=3), frameon=True, loc=2)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        sb.axes_style()
        fig.add_subplot(ax)
        cnt+=1
    df.loc[vsm_cnt] = fscore_row

# save
sb.plt.savefig("../outputs/"+corpus+"_distribs_imposters.pdf")
# clear:
sb.plt.clf()

# take care of the tables:
df = df.set_index("vector space model")
# row and col names:
df.columns.name = "distance metric"
df.index.name = "distance metric"
df.to_csv("../outputs/"+corpus+"_imposters.csv")
print("=== imposter f1-scores ===")
print str(df.to_latex())
