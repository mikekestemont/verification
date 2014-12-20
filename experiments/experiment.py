import logging

from verification.verification import Verification
from verification.evaluation import evaluate, evaluate_with_threshold
from verification.plotting import plot_test_densities, plot_test_results
from verification.preprocessing import prepare_corpus
import seaborn as sb

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

train, dev = "../data/du_essays", "../data/du_essays"
logging.info("preparing corpus")
X_train = prepare_corpus(train)
X_dev = prepare_corpus(dev)
verifier = Verification(random_state=1,
                        metric='minmax', sample_authors=False,
                        n_features=10000,
                        n_test_pairs=10000, em_iterations=10,
                        vector_space_model='tf', weight=0.01,
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

sb.plt.plot(test_recalls, test_precisions)
print "F: %.3f, P: %.3f, R: %.3f" % (test_f, test_p, test_r)

#plot_test_densities(results=results, dev_t=best_t)
#plot_test_results(test_fscores[:-1], test_thresholds, test_precisions[:-1], test_recalls[:-1], best_t)
