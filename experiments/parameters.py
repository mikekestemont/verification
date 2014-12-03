import logging

from itertools import product
from operator import mul

import numpy as np
from joblib import Parallel, delayed

from verification.verification import Verification, evaluate_predictions

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
                    level=logging.INFO)

parameters = {
    'n_features': [500, 1000, 2000, 5000, 10000, 50000, 100000, 500000],
    'random_prop': [0.2, 0.5],
    'metric': ['minmax', 'euclidean', 'cityblock', 'divergence', 'cosine'],
    'vector_space_model': ['std', 'plm', 'idf', 'tf'],
    'weight': np.arange(0.001, 1.001, 0.001),
    'em_iterations': [10, 50, 100]
}


def param_iter(parameters):
    keys, values = zip(*sorted(parameters.items()))
    num_settings = reduce(mul, (len(v) for v in parameters.values()))
    i = 0
    for v in product(*values):
        logging.info("Processing parameter setting %s / %s" % (i, num_settings))
        yield dict(zip(keys, v))
        i += 1

def run_experiment(parameters, X_train, X_dev):
    verification = Verification(**parameters)
    verification.fit(X_train, X_dev)
    results = list(verification.verify())
    predictions, dev_t = evaluate_predictions(results)
    return (next((f1, p, r) for f1, t, p, r in predictions if t == dev_t),
            parameters)

results = Parallel(n_jobs=20)(
    delayed(run_experiment)(params) for params in param_iter(parameters))

with open("results.txt", "w") as outfile:
    for result, params in results:
        outfile.write("%s\t%s\n" % (
            '\t'.join(map(str, result)),
            '\t'.join(value for _, value in sorted(params.items()))))
