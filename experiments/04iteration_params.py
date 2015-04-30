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
n_dev_pairs = 500
random_state = 1000

# select a data set
data_dir = "../data/"
corpus = "gr_articles"
print corpus

# we prepare the corpus
logging.info("preparing corpus")

X_dev, X_test = split_corpus(prepare_corpus(data_dir+corpus), controlled="authors", random_state=random_state)

iteration_ranges = (1, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 300)
prop_ranges = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


df_dev = pd.DataFrame(columns=["nb_iterations"]+[str(n) for n in prop_ranges])

for i, iteration in enumerate(iteration_ranges):
    dev_row = [str(iteration)]
    print "* nr of sampling iterations: "+str(iteration)
    for prop in prop_ranges:
        print "\t+ sampling proportion: "+str(prop)
        verifier = Verification(n_features=mfw,
                                feature_type="words",
                                random_prop=prop,
                                sample_features=True,
                                sample_authors=True,
                                metric=dm,
                                text_cutoff=None,
                                sample_iterations=iteration,
                                n_potential_imposters=60,
                                n_actual_imposters=10,
                                n_test_pairs=0,
                                n_dev_pairs=n_dev_pairs,
                                random_state=random_state,
                                top_rank=10,
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
df_dev.to_csv("../outputs/"+corpus+"_heatmap_sampling.csv")

df_dev = df_dev.set_index("nb_iterations")
df_dev.columns.name = "proportion"
df_dev.index.name = "nb_iterations"
#df_dev = df_dev.applymap(lambda x:int(x*1000))

sb.heatmap(df_dev, annot=True, fmt='.2g')
sb.plt.savefig("../outputs/"+corpus+"_heatmap_sampling.pdf")
sb.plt.clf()
print "=== dev score table ==="
print str(df_dev.to_string())
