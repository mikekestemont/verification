import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.WARN)

import sys
from itertools import product
from operator import mul

import numpy as np
from joblib import Parallel, delayed

from verification.verification import Verification, evaluate_predictions
from verification.preprocessing import prepare_corpus


parameters = {
    'n_features': [50, 500, 1000, 5000, 10000, 50000, 100000, 500000],
    'metric': ['minmax', 'euclidean', 'cityblock', 'divergence', 'cosine'],
    'vector_space_model': ['std', 'plm', 'idf', 'tf'],
    'weight': np.arange(0.001, 1.001, 0.01),
    'em_iterations': [10, 50],
    'random_state': [2014]
}

def param_iter(parameters):
    keys, values = zip(*sorted(parameters.items()))
    for v in product(*values):
        yield dict(zip(keys, v))


def run_experiment(parameters, X_train, X_dev):
    verification = Verification(**parameters)
    verification.fit(X_train, X_dev)
    results = list(verification.verify())
    dev_results = results[:int(len(results) / 2.0)]
    test_results = results[int(len(results) / 2.0):]
    dev_predictions, dev_t, _, _ = evaluate_predictions(dev_results)
    test_predictions = evaluate_predictions(test_results, t=dev_t)
    return test_predictions, parameters

X_train = prepare_corpus(sys.argv[1])
X_dev = prepare_corpus(sys.argv[2])
results = Parallel(n_jobs=20, verbose=1)(
    delayed(run_experiment)(params, X_train, X_dev) for params in param_iter(parameters))

with open("results.txt", "a") as outfile:
    for result, params in results:
        outfile.write("%s\t%s\n" % (
            '\t'.join(map(str, result)),
            '\t'.join(str(value) for _, value in sorted(params.items()))))
