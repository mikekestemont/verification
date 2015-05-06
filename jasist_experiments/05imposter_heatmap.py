"""Visualize effect of potential and actual impostors on modern data. Used to generate Fig. 4"""

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.WARNING)

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
from verification.preprocessing import split_corpus, prepare_corpus
import numpy as np
import pandas as pd

# set parameters:
vsm = 'std'
dm  = 'minmax'
mfw = 10000
intervals = 8
n_dev_pairs = 500
random_state = 1000

# select a data set
data_path = "../data/"
corpus = "gr_articles"
print corpus

# we prepare the corpus
logging.info("preparing corpus")

X_dev, X_test = split_corpus(prepare_corpus(data_path+corpus), controlled="authors", random_state=random_state)

potential_imposter_ranges = [int(i) for i in np.linspace(10, 200, intervals)]
df_dev = pd.DataFrame(columns=["potential"]+[str(n+1) for n in range(intervals)])

for i, n_potential_imposters in enumerate(potential_imposter_ranges):
    dev_row = [str(n_potential_imposters)]
    print "* nr of potential imposters: "+str(n_potential_imposters)
    n_actual_imposter_ranges = [int(i) for i in np.linspace(3, n_potential_imposters, intervals)]
    for n_actual_imposters in n_actual_imposter_ranges:
        print "\t+ nr of actual imposters: "+str(n_actual_imposters)
        verifier = Verification(n_features=mfw,
                                feature_type="words",
                                random_prop=0.5,
                                sample_features=True,
                                sample_authors=True,
                                metric=dm,
                                text_cutoff=None,
                                sample_iterations=10,
                                n_potential_imposters=n_potential_imposters,
                                n_actual_imposters=n_actual_imposters,
                                n_test_pairs=0,
                                n_dev_pairs=n_dev_pairs,
                                random_state=random_state,
                                top_rank=n_actual_imposters,
                                vector_space_model=vsm,
                                balanced_pairs=True)
        logging.info("Starting verification [dev / test]")
        verifier.vectorize(X_dev, X_test)
        dev_results = verifier.fit()
        logging.info("Computing results")
        # get dev results:
        dev_f, dev_p, dev_r, dev_t = evaluate(dev_results)
        max_f = np.nanmax(dev_f)
        dev_row.append(max_f)

    # update dev df:
    df_dev.loc[i] = dev_row
    print "=== dev scores ==="
    print df_dev.to_string()

# process and plot train df:
df_dev.to_csv("../outputs/"+corpus+"_heatmap_imposters.csv")

df_dev = df_dev.set_index("potential")
df_dev.columns.name = "actual"
df_dev.index.name = "potential"
#df_dev = df_dev.applymap(lambda x:int(x*1000))

sb.heatmap(df_dev, annot=True, square=True, fmt='.2g')
sb.plt.savefig("../outputs/"+corpus+"_heatmap_imposters.pdf")
sb.plt.clf()
print "=== dev score table ==="
print str(df_dev.to_string())
