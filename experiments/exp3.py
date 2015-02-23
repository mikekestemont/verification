
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import seaborn as sb

from verification.verification import Verification
from verification.evaluation import evaluate, evaluate_with_threshold, average_precision_score
from verification.evaluation import rank_predict
from verification.plotting import plot_test_results
from verification.preprocessing import prepare_corpus, Dataset
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

# select a data set
train = "../data/du_essays"
test = train
print "Using data under: "+train

# we prepare the corpus
logging.info("preparing corpus")
data = prepare_corpus(train)
train_texts, test_texts, train_titles, test_titles, train_authors, test_authors = train_test_split(
    data.texts, data.titles, data.authors, test_size=0.5, random_state=1000)
X_train = Dataset(train_texts, train_titles, train_authors)
X_test = Dataset(test_texts, test_titles, test_authors)

# we determine the size of the entire vocabulary
V = len(set(sum(X_train.texts, []) + sum(X_test.texts, [])))

#vsms = ('std', 'plm', 'tf', 'idf')
#dms  = ('cosine', 'euclidean', 'cityblock', 'divergence', 'minmax')

vsm = 'plm'
dm  = 'minmax'
print "\t* "+vsm+" & "+dm

print "=== BASELINE (without sampling) ==="
verifier = Verification(n_features=2400,
                        random_prop=0.5,
                        sample_features=False,
                        sample_authors=False,
                        metric=dm,
                        text_cutoff=None,
                        sample_iterations=100,
                        n_potential_imposters=100,
                        n_actual_imposters=25,
                        n_train_pairs=500,
                        n_test_pairs=500,
                        random_state=1000,
                        vector_space_model=vsm,
                        balanced_pairs=True)
logging.info("Starting verification [train / test]")
verifier.fit(X_train, X_test)
train_results, test_results = verifier.verify()
logging.info("Computing results")
dev_f, dev_p, dev_r, dev_t = evaluate(train_results)
best_t = dev_t[np.nanargmax(dev_f)]
test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t)
print "\t\t- F-score: "+str(test_f)
print "\t\t- Precision: "+str(test_p)
print "\t\t- Recall: "+str(test_r)

intervals = 8

print "=== RESAMPLING ==="
potential_imposter_ranges = [int(i) for i in np.linspace(10, int(len(train_titles)/2), intervals)]
df = pd.DataFrame(columns=["potential"]+[str(n+1) for n in range(intervals)])
print df.to_string()
for i, n_potential_imposters in enumerate(potential_imposter_ranges):
    row = [str(n_potential_imposters)]
    print "* nr of potential imposters: "+str(n_potential_imposters)
    n_actual_imposter_ranges = [int(i) for i in np.linspace(1, n_potential_imposters, intervals)]
    for n_actual_imposters in n_actual_imposter_ranges:
        print "\t+ nr of actual imposters: "+str(n_actual_imposters)
        verifier = Verification(n_features=V,
                                random_prop=0.5,
                                sample_features=True,
                                sample_authors=True,
                                metric=dm,
                                text_cutoff=None,
                                sample_iterations=100,
                                n_potential_imposters=n_potential_imposters,
                                n_actual_imposters=n_actual_imposters,
                                n_train_pairs=500,
                                n_test_pairs=500,
                                random_state=1,
                                top_rank=1,
                                vector_space_model=vsm,
                                balanced_pairs=True)
        logging.info("Starting verification [train / test]")
        verifier.fit(X_train, X_test)
        train_results, test_results = verifier.verify()
        logging.info("Computing results")
        dev_f, dev_p, dev_r, dev_t = evaluate(train_results)
        best_t = dev_t[np.nanargmax(dev_f)]
        test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t)
        row.append(test_f)
        print "\t\t- F-score: "+str(test_f)
        print "\t\t- Precision: "+str(test_p)
        print "\t\t- Recall: "+str(test_r)
    df.loc[i] = row
    print df.to_string()
df = df.set_index("potential")
df.columns.name = "actual imposters"
df.index.name = "potential imposters"
df = df.applymap(lambda x:int(x*100))
sb.plt.figure()
sb.heatmap(df, annot=True)
sb.plt.savefig("../plots/exp3.pdf")
sb.plt.clf()
print str(df.to_string())