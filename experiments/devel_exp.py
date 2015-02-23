"""
We plot the different distributions in scores for same-author and different-author pairs
in the dev set, and calculate whether they are statistically significantly different,
using the entire vocabulary.
"""

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
from verification.plotting import draw_tree
from verification.preprocessing import prepare_corpus, Dataset
from sklearn.cross_validation import train_test_split
import numpy as np

# select a data set
corpus = "../data/du_essays"

# we prepare the corpus
logging.info("preparing corpus")
data = prepare_corpus(corpus)
train_texts, test_texts, train_titles, test_titles, train_authors, test_authors = train_test_split(
      data.texts, data.titles, data.authors, test_size=0.5, random_state=1000)
X_train = Dataset(train_texts, train_titles, train_authors)
X_test = Dataset(test_texts, test_titles, test_authors)

# we determine the size of the entire vocabulary
V = 2000
vsm = 'plm'
dm  = 'minmax'

verifier = Verification(random_state=1000,
                        sample_features=False,
                        metric=dm,
                        sample_authors=False,
                        n_features=V,
                        n_train_pairs=100,
                        n_test_pairs=100,
                        em_iterations=100,
                        vector_space_model=vsm,
                        weight=0.2,
                        n_actual_imposters=10,
                        eps=0.01,
                        norm="l2",
                        top_rank=1,
                        balanced_pairs=True)

logging.info("Starting verification [train / test]")
verifier.fit(X_train, X_test)
train_results, test_results = verifier.verify()
logging.info("Computing results")
train_f, train_p, train_r, train_t = evaluate(train_results)

best_train_t = train_t[np.nanargmax(train_f)]
test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_train_t)

print("Test results for: "+vsm+" & "+dm+":")
print("\t\t- F-score: "+str(test_f))
print("\t\t- Precision: "+str(test_p))
print("\t\t- Recall: "+str(test_r))
