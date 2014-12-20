import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.WARN)

import sys
from itertools import product
from operator import mul

import numpy as np
from joblib import Parallel, delayed

from verification.verification import Verification, evaluate, evaluate_with_threshold
from verification.preprocessing import prepare_corpus


parameters = {
    'n_features': np.arange(500, 50000, 500),
    'metric': ['euclidean', 'minmax', 'cityblock', 'divergence', 'cosine'],
    'vector_space_model': ['plm', 'std', 'idf', 'tf'],
    'weight': [0.05],
    'em_iterations': [50],
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
    dev_f, dev_p, dev_r, dev_t = evaluate(dev_results)
    best_t = dev_t[dev_f.argmax()]
    test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t)
    return [test_f, test_p, test_r], parameters

X_train = prepare_corpus(sys.argv[1])
X_dev = prepare_corpus(sys.argv[2])
results = Parallel(n_jobs=12, verbose=5)(
    delayed(run_experiment)(params, X_train, X_dev) for params in param_iter(parameters))
#results = [run_experiment(params, X_train, X_dev) for params in param_iter(parameters)]

with open(sys.argv[3], "w") as outfile:
    for result, params in results:
        outfile.write("%s\t%s\n" % (
            '\t'.join(map(str, result)),
            '\t'.join(str(value) for _, value in sorted(params.items()))))
