""" Experiment 1: Baseline experiment. """

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.ERROR)

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
corpora = ["du_essays", "gr_articles", "caesar_background", "sp_articles"]
#corpora = ["du_essays"]
n_experiments = 100

corpora_results = {}

for corpus in corpora:
    print("=== "+corpus+" ===")
    # we select a data set and a distance metric:
    train = data_path+corpus
    test = train
    # we prepare the corpus:
    logging.info("preparing corpus")
    data = prepare_corpus(train)
    train_texts, test_texts, train_titles, test_titles, train_authors, test_authors = train_test_split(
        data.texts, data.titles, data.authors, test_size=0.5, random_state=1000)
    X_train = Dataset(train_texts, train_titles, train_authors)
    X_test = Dataset(test_texts, test_titles, test_authors)

    # we determine the size of the vocabulary
    V = len(set(sum(X_train.texts, []) + sum(X_test.texts, [])))
    # we define the intervals which which to increase the top-n features (MFW)
    feature_ranges = [int(x) for x in np.linspace(30, V, n_experiments)]

    vsms = ('std', 'plm', 'tf', 'idf')

    f1_df = pd.DataFrame(columns=["distance_metric"]+list(vsms))
    nf_df = pd.DataFrame(columns=["distance_metric"]+list(vsms))
    # we iterate over the distance metrics:
    for i, distance_metric in enumerate(['minmax', 'divergence', 'euclidean', 'cityblock']):
        print("* "+distance_metric)
        # we iterate over the vector space models:
        vsm_fscore_row = [distance_metric]
        vsm_nfeat_row = [distance_metric]
        for vsm in vsms:
            print("\t+ "+vsm)
            train_f_scores, test_f_scores = [], []
            for n_features in feature_ranges:
                verifier = Verification(random_state=1000,
                                        metric=distance_metric,
                                        sample_authors=False,
                                        sample_features=False,
                                        n_features=n_features,
                                        n_test_pairs=500,
                                        n_train_pairs=500,
                                        em_iterations=100,
                                        vector_space_model=vsm,
                                        weight=0.2,
                                        top_rank=1,
                                        eps=0.01,
                                        norm="l2",
                                        balanced_pairs=True)
                logging.info("Starting verification [train / test]")
                verifier.fit(X_train, X_test)
                results, test_results = verifier.verify()

                logging.info("Computing results")
                train_f, train_p, train_r, train_t = evaluate(results)

                best_t = train_t[np.nanargmax(train_f)]
                train_f_scores.append(np.nanmax(train_f))

                test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t)
                test_f_scores.append(test_f)

            # collect max_score across feature ranges:
            best_index = np.nanargmax(train_f_scores)
            best_n_features = feature_ranges[best_index]
            vsm_nfeat_row.append(best_n_features)
            print("\t\tbest n_features: "+str(best_n_features))
            # collect test results:
            f_test = test_f_scores[best_index]
            vsm_fscore_row.append(f_test)
            print("\t\tF1-score: "+str(f_test))
            # plot the results
            sb.set_style("darkgrid")
            sb.plt.plot(feature_ranges, train_f_scores, label=vsm)

        f1_df.loc[i] = vsm_fscore_row
        nf_df.loc[i] = vsm_nfeat_row
        # plot the results:
        sb.plt.title(distance_metric)
        sb.plt.legend(loc='best')
        sb.plt.savefig("../plots/exp1_"+distance_metric+".pdf")
        sb.plt.clf()
    # set indices:
    f1_df = f1_df.set_index("distance_metric")
    nf_df = nf_df.set_index("distance_metric")
    # row and col names:
    f1_df.columns.name = "vector space model"
    nf_df.columns.name = "vector space model"
    f1_df.index.name = "distance metric"
    nf_df.index.name = "distance metric"
    # plot fscores:
    corpora_results[corpus+"_f-scores"] = f1_df
    corpora_results[corpus+"_n-features"] = nf_df
    print("=== f-scores ===")
    print str(f1_df.to_latex())
    print("=== n-features ===")
    print str(nf_df.to_latex())
