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
from verification.preprocessing import prepare_corpus, split_corpus, ngram_analyzer

random_state = 1000
data_path = "../data/"
corpus = "caesar_dev"
print corpus

n_experiments = 50
n_dev_pairs = 500

logging.info("preparing corpus")
X_dev, X_test = split_corpus(prepare_corpus(data_path+corpus), controlled="authors", random_state=random_state)

vsms = ('std', 'tf', 'tfidf')
dms = ('minmax', 'euclidean', 'cityblock')
ftypes = ('chars', 'words')

for ftype in ftypes:
    if ftype == "words":
        V = int(len(set(sum(X_dev.texts, []) + sum(X_test.texts, [])))/2.0)
    elif ftype == "chars":
        unique_ngrams = set(ngram_analyzer(sum(X_dev.texts, []) + sum(X_test.texts, []), n=4))
        V = int(len(unique_ngrams)/2.0)
    feature_ranges = [int(x) for x in np.linspace(50, V, n_experiments)]
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
                                        feature_type=ftype,
                                        sample_authors=False,
                                        sample_features=False,
                                        n_features=n_features,
                                        n_test_pairs=0,
                                        n_dev_pairs=n_dev_pairs,
                                        vector_space_model=vsm,
                                        ngram_range=4,
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
        sb.plt.savefig("../outputs/"+corpus+"_featrngs_"+dm+"_baseline_"+ftype+".pdf")
        sb.plt.clf()

