from __future__ import print_function

from collections import Counter
import logging
import random
from operator import itemgetter
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
from verification.preprocessing import prepare_corpus, Dataset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans

from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, show, output_file, save
from bokeh.charts import Bar
from bokeh.io import output_file, show, vplot, save
from bokeh.plotting import figure
from bokeh.models import Axis

data_path = '../data/'
corpus = 'soldier_letters'

# we first prepare the corpus in the normal way:
verif_dataset = prepare_corpus(data_path+corpus)

# we check which scribes appear more than once:
cnt = Counter(verif_dataset.authors)
included_authors = set(a for a in cnt if cnt[a] > 1)

# now, we jumble the words in each letter and
# divide them into two halves (for development purposes):
lookup = [] # for later convenience
lookup_idx = 0
random.seed(1072015)
texts, titles, authors = [], [], []
for text, title, author in zip(verif_dataset.texts, verif_dataset.titles, verif_dataset.authors):
    if author in included_authors:
        random.shuffle(text)
        text_a, text_b = text[:int(len(text)/2.0)], text[int(len(text)/2.0):]
        code = str(lookup_idx)
        # first half:
        texts.append(text_a)
        titles.append(code+'a')
        authors.append(code)
        # second half:
        texts.append(text_b)
        titles.append(code+'b')
        authors.append(code)
        lookup.append((author, title))
        lookup_idx += 1

test_dataset = Dataset(texts, titles, authors)



vsms = ('std', 'tf', 'tfidf')
dms = ('minmax', 'euclidean', 'cityblock')
random_state = 999
n_features = 10000
balanced_pairs = False
n_dev_pairs = 1000

#### first, we try out different combinations of vector spaces and distance metrics:
# set fig params
fig = sb.plt.figure(figsize=(len(vsms), len(dms)))
cnt = 0
outer_grid = gridspec.GridSpec(len(vsms), len(dms), wspace=0.1, hspace=0.1)
c1, c2 = sb.color_palette('Set1')[:2]


# first baseline:
df = pd.DataFrame(columns=['vector space model']+list(dms))

for vsm_cnt, vsm in enumerate(vsms):
    print('\t+ '+vsm)
    fscore_row = [vsm]
    for dm_cnt, dm in enumerate(dms):
        print('\t\t* '+dm)
        verifier = Verification(random_state=random_state,
                                metric=dm,
                                feature_type='chars',
                                ngram_range=4,
                                sample_authors=False,
                                sample_features=False,
                                n_features=n_features,
                                n_test_pairs=None,
                                n_dev_pairs=n_dev_pairs,
                                vector_space_model=vsm,
                                balanced_pairs=balanced_pairs)
        logging.info("Starting verification [dev / test]")
        verifier.vectorize(test_dataset)
        dev_results = verifier.fit()
        logging.info("Computing results")
        dev_f, dev_p, dev_r, dev_t = evaluate(dev_results)
        max_f = np.nanmax(dev_f)
        print('\t\t + F1: ', max_f)
        fscore_row.append(format(max_f*100, '.1f'))

        # distribution of scores:
        same_author_densities = np.asarray([sc for c, sc in dev_results if c == "same_author"])
        diff_author_densities = np.asarray([sc for c, sc in dev_results if c == "diff_author"])

        D, p = ks_2samp(same_author_densities, diff_author_densities)
        print("\t\t- KS: D = %s (p = %s)" %(D, p))
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
sb.plt.savefig('outputs/distribs_baseline.pdf')

# take care of the tables:
df = df.set_index('vector space model')
# row and col names:
df.columns.name = "vector space"
df.index.name = "distance metric"
df.to_csv('outputs/baseline.csv')
print("=== baseline f1-scores ===")
print(str(df.to_latex()))

#### now, we check different n_features and ngram_ranges (for minmax and std):
V = int(len(set(sum(test_dataset.texts, []) + sum(test_dataset.texts, []))))
print('vocab size: ', V)
ngram_sizes = [2, 3, 4, 5]
feature_ranges = [int(x) for x in np.linspace(30, V, 100)]
print(feature_ranges)

# first baseline:
f, ax = plt.subplots(1,1)
sb.set_style("darkgrid")
ax.set_ylim(.5, 1)
for ns in ngram_sizes:
    print('* ngram size:', ns)
    f_scores = []
    for n_features in feature_ranges:
        print('\t\t* Testing nr features:', n_features)
        verifier = Verification(random_state=random_state,
                                metric='minmax',
                                feature_type='chars',
                                ngram_range=ns,
                                n_features=n_features,
                                n_test_pairs=0,
                                n_dev_pairs=n_dev_pairs,
                                vector_space_model='std',
                                balanced_pairs=balanced_pairs)
        logging.info("Starting verification [dev / test]")
        verifier.vectorize(test_dataset)
        dev_results = verifier.fit()
        logging.info("Computing results")
        dev_f, dev_p, dev_r, dev_t = evaluate(dev_results)
        max_f = np.nanmax(dev_f)
        print('\t\t + F1:', max_f)
        f_scores.append(max_f)
    sb.plt.plot(feature_ranges, f_scores, label=str(ns))

# plot the results:
sb.plt.title('Effect of # features and size of the ngrams')
sb.plt.legend(loc='best')
sb.plt.savefig('outputs/featrngs_baseline.pdf')
sb.plt.clf()
# save
sb.plt.savefig('outputs/distribs_baseline.pdf')


#####################################################################
### now imposters ###################################################
#####################################################################


for bg in ('soldier_armen', 'soldier_sailing', 'soldier_letters'):
    # set fig params
    sb.plt.clf()
    fig = sb.plt.figure(figsize=(len(vsms), len(dms)))
    cnt = 0
    outer_grid = gridspec.GridSpec(len(vsms), len(dms), wspace=0.1, hspace=0.1)
    c1, c2 = sb.color_palette("Set1")[:2]
    background_dataset = None
    if background_dataset != 'soldier_letters':
        background_dataset = prepare_corpus(data_path+bg)

    df = pd.DataFrame(columns=["vector space model"]+list(dms))

    for vsm_cnt, vsm in enumerate(vsms):
        print("\t+ "+vsm)
        fscore_row = [vsm]
        for dm_cnt, dm in enumerate(dms):
            print("* "+dm)
            verifier = Verification(random_state=random_state,
                                    metric=dm,
                                    feature_type='chars',
                                    ngram_range=4,
                                    sample_authors=True,
                                    sample_iterations=100,
                                    sample_features=True,
                                    n_features=10000,
                                    random_prop=0.5,
                                    n_test_pairs=0,
                                    n_dev_pairs=n_dev_pairs,
                                    vector_space_model=vsm,
                                    n_potential_imposters=50,
                                    n_actual_imposters=10,
                                    top_rank=10,
                                    balanced_pairs=balanced_pairs)
            logging.info("Starting verification [dev / test]")
            if background_dataset:
                verifier.vectorize(test_dataset, background_dataset)
            else:
                verifier.vectorize(test_dataset)
            dev_results = verifier.fit(filter_imposters=True)

            logging.info("Computing results")
            dev_f, dev_p, dev_r, dev_t = evaluate(dev_results)
            max_f = np.nanmax(dev_f)
            print('\t\t + F1:', max_f)
            fscore_row.append(format(max_f*100, '.1f'))

            # distribution of scores:
            same_author_densities = np.asarray([sc for c, sc in dev_results if c == "same_author"])
            diff_author_densities = np.asarray([sc for c, sc in dev_results if c == "diff_author"])

            D, p = ks_2samp(same_author_densities, diff_author_densities)
            print('\t\t\t- KS: D =', D, '(p = ', p, ')')
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
    sb.plt.savefig("outputs/distr_impost(%s_as_background).pdf" %(bg))
    # clear:
    sb.plt.clf()

    # take care of the tables:
    df = df.set_index("vector space model")
    # row and col names:
    df.columns.name = "vector space model"
    df.index.name = "distance metric"
    df.to_csv("../outputs/sc_imposters(%s_as_background).csv" %(bg))
    print("=== imposter f1-scores ===")
    print(str(df.to_latex()))
