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
corpus = "caesar_dev"
n_experiments = 10
n_dev_pairs = 500
n_test_pairs = n_dev_pairs

print("=== "+corpus+" ===")
dev = data_path+corpus
logging.info("preparing corpus")
X_dev, X_test = split_corpus(prepare_corpus(dev), controlled="authors", random_state=random_state)
V = int(len(set(sum(X_dev.texts, []) + sum(X_test.texts, [])))/2.0)
feature_ranges = [int(x) for x in np.linspace(50, V, n_experiments)]
print feature_ranges
vsms = ('std', 'tf', 'tfidf', 'bin', 'plm')
f1_df = pd.DataFrame(columns=["distance_metric"]+list(vsms))
nf_df = pd.DataFrame(columns=["distance_metric"]+list(vsms))
for i, distance_metric in enumerate(['minmax', 'euclidean', 'cityblock']):
    print("* "+distance_metric)
    f, ax = plt.subplots(1,1)
    sb.set_style("darkgrid")
    ax.set_ylim(.5, 1)
    # we iterate over the vector space models:
    vsm_fscore_row = [distance_metric]
    vsm_nfeat_row = [distance_metric]
    for vsm in vsms:
        print("\t+ "+vsm)
        dev_f_scores = []
        for n_features in feature_ranges:
            print "\t\t* Testing nr features: "+str(n_features)
            verifier = Verification(random_state=random_state,
                                    metric=distance_metric,
                                    sample_authors=False,
                                    sample_features=False,
                                    n_features=n_features,
                                    n_test_pairs=n_test_pairs,
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
            max_f1 = np.nanmax(dev_f)
            print "\t\t + F1: "+str(max_f1)
            dev_f_scores.append(max_f1)
        sb.plt.plot(feature_ranges, dev_f_scores, label=vsm)
        # find optimum in smoothed scores:
        model = SuperSmoother()
        model.fit(feature_ranges, dev_f_scores)
        smoothed_f1s = model.predict(feature_ranges)
        best_n_features = feature_ranges[np.nanargmax(smoothed_f1s)]
        vsm_nfeat_row.append(best_n_features)
        print("\t\tbest n_features: "+str(best_n_features))
        # refit the model:
        verifier = Verification(random_state=random_state,
                                metric=distance_metric,
                                sample_authors=False,
                                sample_features=False,
                                n_features=best_n_features,
                                n_test_pairs=n_test_pairs,
                                n_dev_pairs=n_dev_pairs,
                                em_iterations=100,
                                vector_space_model=vsm,
                                weight=0.2,
                                eps=0.01,
                                norm="l2",
                                balanced_pairs=True)
        verifier.vectorize(X_dev, X_test)
        dev_results, test_results = verifier.predict()
        dev_f, dev_p, dev_r, dev_t = evaluate(dev_results)
        best_t = dev_t[np.nanargmax(dev_f)]
        test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t)
        vsm_fscore_row.append(test_f)
        print("\t\tF1-score: "+str(test_f))
    f1_df.loc[i] = vsm_fscore_row
    nf_df.loc[i] = vsm_nfeat_row
    # plot the results:
    sb.plt.title(distance_metric)
    sb.plt.legend(loc='best')
    sb.plt.savefig("../plots/exp1_"+distance_metric+"_"+corpus+".pdf")
    sb.plt.clf()
# set indices:
f1_df = f1_df.set_index("distance_metric")
nf_df = nf_df.set_index("distance_metric")
# row and col names:
f1_df.columns.name = "vector space model"
nf_df.columns.name = "vector space model"
f1_df.index.name = "distance metric"
nf_df.index.name = "distance metric"
nf_df.to_csv("../plots/exp1_"+distance_metric+"_"+corpus+"_nf.csv")
f1_df.to_csv("../plots/exp1_"+distance_metric+"_"+corpus+"_f1.csv")
print("=== f-scores ===")
print str(f1_df.to_latex())
print("=== n-features ===")
print str(nf_df.to_latex())
