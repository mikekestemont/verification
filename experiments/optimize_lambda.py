import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.WARN)

import numpy as np
import pandas as pd
from verification.verification import Verification
from verification.evaluation import evaluate, evaluate_with_threshold
from verification.plotting import plot_test_densities, plot_test_results
from sklearn.preprocessing import MinMaxScaler
from verification.preprocessing import prepare_corpus
import seaborn as sb
from joblib import Parallel, delayed



train, dev = "../data/du_essays", "../data/du_essays"
logging.info("preparing corpus")
X_train = prepare_corpus(train)
X_dev = prepare_corpus(dev)

all_scores = []
metric = 'cosine'
lambdas = np.arange(0.001, 1.0, 0.01)
n_feature_ranges = np.arange(50, 10000, 250)
combinations = lambdas.shape[0] * n_feature_ranges.shape[0]

def experiment(Lambda):
    df = pd.DataFrame(columns=['Lambda', 'n_features', 'F', 'P', 'R'])
    for k, n_features in enumerate(n_feature_ranges):
        verifier = Verification(random_state=1,
                                metric=metric, sample_authors=False,
                                n_features=n_features,
                                n_test_pairs=10000, em_iterations=10,
                                vector_space_model='plm', weight=Lambda,
                                n_actual_imposters=10,
                                top_rank=10)
        logging.info("Starting verification")
        verifier.fit(X_train, X_dev)
        results = list(verifier.verify())
        logging.info("Computing results")
        dev_results = results[:int(len(results) / 2.0)]
        test_results = results[int(len(results) / 2.0):]
        dev_f, dev_p, dev_r, dev_t = evaluate(dev_results)
        best_t = dev_t[dev_f.argmax()]
        test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t)
        test_fscores, test_precisions, test_recalls, test_thresholds = evaluate(test_results)
        df.loc[k] = np.array([Lambda, n_features, test_f, test_p, test_r])
    return df

all_scores = Parallel(n_jobs=8, verbose=5)(
    delayed(experiment)(Lambda) for Lambda in lambdas)
df = pd.concat(all_scores)
df[['n_features', 'F', 'P', 'R']] = df[['n_features', 'F', 'P', 'R']].astype(np.float64)
for key, group in df.groupby('Lambda'):
    sb.plt.plot(group.n_features, group.F, label=key)
sb.plt.legend(loc='best')
sb.plt.xlabel("$n$ features")
sb.plt.ylabel("$F1$ score")
sb.plt.title(metric)
#scaler = MinMaxScaler()
#df.F = scaler.fit_transform(df.F)
#sb.plt.plot(test_recalls, test_precisions)
#print "F: %.3f, P: %.3f, R: %.3f" % (test_f, test_p, test_r)
