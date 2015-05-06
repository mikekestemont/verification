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

from verification.verification import Verification
from verification.evaluation import evaluate, evaluate_with_threshold
from verification.preprocessing import prepare_corpus, split_corpus

from supersmoother import SuperSmoother

random_state = 1000
data_path = "../data/"
corpus = "gr_articles"
print corpus

n_experiments = 50
n_dev_pairs = 500

logging.info("preparing corpus")
X_dev, X_test = split_corpus(prepare_corpus(data_path+corpus), controlled="authors", random_state=random_state)

V = int(len(set(sum(X_dev.texts, []) + sum(X_test.texts, [])))/2.0)
feature_ranges = [int(x) for x in np.linspace(50, V, n_experiments)]
print feature_ranges

vsms = ('std', 'tf', 'tfidf', 'bin', 'plm')
dms = ('minmax', 'euclidean', 'cityblock')

# first baseline:
for i, dm in enumerate(dms):
    print("* "+dm)
    f, ax = plt.subplots(1,1)
    sb.set_style("darkgrid")
    ax.set_ylim(.5, 1)
    for vsm in vsms:
        print("\t+ "+vsm)
        f_scores = []
        for n_features in feature_ranges:
            print "\t\t* Testing nr features: "+str(n_features)
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
            f_scores.append(max_f)
        sb.plt.plot(feature_ranges, f_scores, label=vsm)

    # plot the results:
    sb.plt.title(dm)
    sb.plt.legend(loc='best')
    sb.plt.savefig("../outputs/"+corpus+"_featrngs_"+dm+"_baseline.pdf")
    sb.plt.clf()

# secondly imposters:
for i, dm in enumerate(dms):
    print("* "+dm)
    f, ax = plt.subplots(1,1)
    sb.set_style("darkgrid")
    ax.set_ylim(.5, 1)
    for vsm in vsms:
        print("\t+ "+vsm)
        f_scores = []
        for n_features in feature_ranges:
            print "\t\t* Testing nr features: "+str(n_features)
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
                                    n_potential_imposters=60,
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
            f_scores.append(max_f)
        sb.plt.plot(feature_ranges, f_scores, label=vsm)

    # plot the results:
    sb.plt.title(dm)
    sb.plt.legend(loc='best')
    sb.plt.savefig("../outputs/"+corpus+"_featrngs_"+dm+"_imposters.pdf")
    sb.plt.clf()

