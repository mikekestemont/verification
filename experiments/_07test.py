import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.WARNING)

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

random_state = 1000
data_path = "../data/"
corpus = "caesar_dev"
n_pairs = 500
n_features = 10000

logging.info("preparing corpus")
X_dev, X_test = split_corpus(prepare_corpus(data_path+corpus), controlled="authors", random_state=random_state)

verifier = Verification(random_state=random_state,
                        metric="minmax",
                        feature_type="words",
                        sample_authors=False,
                        sample_features=False,
                        n_features=n_features,
                        n_test_pairs=n_pairs,
                        n_dev_pairs=n_pairs,
                        vector_space_model="tf",
                        balanced_pairs=True)
logging.info("Starting verification [dev / test]")
verifier.vectorize(X_dev, X_test)
dev_results, test_results = verifier.predict()
logging.info("Computing results")

# first prec rec curve of test results:
test_Fs, test_Ps, test_Rs, test_Ts = evaluate(test_results)
fig = sb.plt.figure()
sb.plt.xlabel("recall", fontsize=10)
sb.plt.ylabel("precision", fontsize=10)
sb.plt.xlim(0.4, 1)
sb.plt.ylim(0.4, 1.05)
sb.plt.plot(test_Rs, test_Ps, label="pairwise")

# get max for dev:
dev_Fs, dev_Ps, dev_Rs, dev_Ts = evaluate(test_results)
best_t = dev_Ts[np.nanargmax(dev_Fs)]
baseline_test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t)
print "Pairwise F1: "+str(baseline_test_f)

verifier = Verification(random_state=random_state,
                        metric="minmax",
                        feature_type="words",
                        sample_authors=True,
                        sample_iterations=100,
                        sample_features=True,
                        n_features=n_features,
                        random_prop=0.5,
                        n_test_pairs=n_pairs,
                        n_dev_pairs=n_pairs,
                        vector_space_model="tf",
                        n_potential_imposters=100,
                        n_actual_imposters=10,
                        top_rank=10,
                        balanced_pairs=True)
logging.info("Starting verification [dev / test]")
verifier.vectorize(X_dev, X_test)
dev_results, test_results = verifier.predict()
logging.info("Computing results")

# add prec rec curve of imposter approach for test results:
test_Fs, test_Ps, test_Rs, test_Ts = evaluate(test_results)
sb.plt.plot(test_Rs, test_Ps, label="imposters")
sb.plt.legend(loc="best")
sb.plt.gca().set_aspect('equal', adjustable='box')
sb.plt.savefig("../outputs/"+corpus+"_test_prec_rec.pdf")

fig = sb.plt.figure()
sb.set_style("darkgrid")

c1, c2, c3, c4 = sb.color_palette("Set1")[:4]
sb.plt.plot(test_Ts, test_Fs, label="F1 score", c=c1)
sb.plt.plot(test_Ts, test_Ps, label="Precision", c=c2)
sb.plt.plot(test_Ts, test_Rs, label="Recall", c=c3)
sb.plt.xlim(0, 1.005)
sb.plt.ylim(0.4, 1.005)

# get optimal threshold on dev set:
dev_Fs, dev_Ps, dev_Rs, dev_Ts = evaluate(dev_results)
best_t = dev_Ts[np.nanargmax(dev_Fs)]
test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t)
print "Impostors F1: "+str(test_f)

ax = sb.plt.gca()
an1 = ax.annotate("Impostors: "+format(test_f*100, '.1f')+"\nPairwise: "+format(baseline_test_f*100, '.1f'),
                  xy=(best_t, 0.45), xycoords="data",
                  va="center", ha="left", fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.6", fc="w"))

sb.plt.axvline(x=best_t, linewidth=1, c=c4)
sb.plt.legend(loc="best")
sb.plt.xlabel('Threshold', fontsize=10)
sb.plt.ylabel('Score', fontsize=10)
sb.plt.savefig("../outputs/"+corpus+"_test_thresholds.pdf")

