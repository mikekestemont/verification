"""
# First experiment
* Setup: For a given distance metric and data set, we compare the performance across several vector spaces.
We report F1-scores across for an increasing number of MFW (starting a 50).
We use traditional score-thresholding and a 50-50 dev-test split of the available documents (randomly selected 10000 test).
* Observations:
Using a larger vocabulary consistently helps (results quickly max out after 5,000 features).
update > Have checked this for all data sets; same pattern emegeres
Ergo: In the rest of the paper we can use all available features without significantly harming performance.
Std and plm yield the best results. Tf hardly helps. Tf-idf yields surprisingly unstable results.
"""

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.WARN)

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import seaborn as sb

import pandas as pd

from verification.verification import Verification
from verification.evaluation import evaluate, evaluate_with_threshold
from verification.preprocessing import prepare_corpus, Dataset
from sklearn.cross_validation import train_test_split
import numpy as np


data_path = "../data/"
#corpora = ["du_essays", "gr_articles", "caesar_background", "sp_articles"]
corpora = ["du_essays"]
n_experiments = 100

corpora_results = {}

for corpus in corpora:
    # we select a data set and a distance metric:
    train = data_path+corpus
    test = train
    # we prepare the corpus:
    logging.info("preparing corpus")
    if train == test:
        data = prepare_corpus(train)
        train_texts, test_texts, train_titles, test_titles, train_authors, test_authors = train_test_split(
            data.texts, data.titles, data.authors, test_size=0.5, random_state=1000)
        X_train = Dataset(train_texts, train_titles, train_authors)
        X_test = Dataset(test_texts, test_titles, test_authors)
    else:
        X_train = prepare_corpus(train)
        X_test = prepare_corpus(test)

    # we determine the size of the vocabulary
    V = len(set(sum(X_train.texts, []) + sum(X_test.texts, [])))
    # we define the intervals which which to increase the top-n features (MFW)
    feature_ranges = np.linspace(30, V, n_experiments)

    vsms = ('std', 'plm', 'tf', 'idf')

    df = pd.DataFrame(columns=["distance_metric"]+list(vsms))
    # we iterate over the distance metrics:
    for i, distance_metric in enumerate(['minmax', 'divergence', 'euclidean', 'cityblock']):
        # we iterate over the vector space models:
        vsm_row = [distance_metric]
        for vsm in vsms:
            f_scores = []
            for n_features in feature_ranges:
                verifier = Verification(random_state=1,
                                        metric=distance_metric,
                                        sample_authors=False,
                                        sample_features=False,
                                        n_features=int(n_features),
                                        n_test_pairs=10000,
                                        n_train_pairs=10000,
                                        em_iterations=100,
                                        vector_space_model=vsm,
                                        weight=0.2,
                                        eps=0.01,
                                        norm="l2",
                                        balanced_pairs=True)
                logging.info("Starting verification [train / test]")
                verifier.fit(X_train, X_test)
                results, test_results = verifier.verify()

                logging.info("Computing results")
                dev_f, dev_p, dev_r, dev_t = evaluate(results)
                #print np.nanmax(dev_f)
                best_t = dev_t[np.nanargmax(dev_f)]

                test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t) 
                f_scores.append(test_f)

            print vsm, sum(f_scores), sum(f_scores) / len(f_scores)
            # collect max_score across feature ranges
            vsm_row.append(np.nanmax(f_scores))
            # plot the results
            sb.set_style("darkgrid")
            sb.plt.plot(feature_ranges, f_scores, label=vsm)
        print vsm_row
        df.loc[i] = vsm_row
        # plot the results:
        sb.plt.title(distance_metric)
        sb.plt.legend(loc='best')
        sb.plt.savefig("../plots/exp1_"+distance_metric+".pdf")
        sb.plt.clf()
    corpora_results[corpus] = df
    print str(df.to_string())
