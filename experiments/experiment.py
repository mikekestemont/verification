import logging

from verification.verification import Verification, evaluate_predictions
from verification.plotting import prec_recall_curve, plot_test_results
from verification.verification import get_result_for_threshold
from verification.preprocessing import prepare_corpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

train, dev = "../data/du_essays", "../data/du_essays"
logging.info("preparing corpus")
X_train = prepare_corpus(train)
X_dev = prepare_corpus(dev)
verifier = Verification(random_state=1,
                        metric='minmax', sample_authors=True,
                        n_features=10000,
                        n_test_pairs=20000, em_iterations=10,
                        vector_space_model='std', weight=0.01)
logging.info("Starting verification")
verifier.fit(X_train, X_dev)
results = list(verifier.verify())
logging.info("Computing results")
predictions, dev_t = evaluate_predictions(results)
plot_test_results(predictions, dev_t)
print next(f1 for f1, t, _, _ in predictions if t == dev_t)
