import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.WARN)


from verification.verification import Verification
from verification.evaluation import evaluate, evaluate_with_threshold
from verification.preprocessing import prepare_corpus, Dataset
from sklearn.cross_validation import train_test_split
import numpy as np
import seaborn as sb


train, test = "../data/du_essays", "../data/du_essays"
logging.info("preparing corpus")
if train == test:
    data = prepare_corpus(train)
    train_texts, test_texts, train_titles, test_titles, train_authors, test_authors = train_test_split(
        data.texts, data.titles, data.authors, test_size=0.5, random_state=1000)
    X_train = Dataset(train_texts.tolist(), train_titles.tolist(), train_authors.tolist())
    X_test = Dataset(test_texts.tolist(), test_titles.tolist(), test_authors.tolist())
else:
    X_train = prepare_corpus(train)
    X_test = prepare_corpus(test)

V = len(set(sum(X_train.texts, []) + sum(X_test.texts, [])))
for vector_space_model in ('std', 'plm', 'tf', 'idf'):
    f_scores = []
    feature_ranges = np.linspace(50, V, 20)
    for n_features in feature_ranges:
        verifier = Verification(random_state=1,
                                metric='euclidean', sample_authors=False,
                                n_features=int(n_features),
                                n_test_pairs=10000, em_iterations=100,
                                vector_space_model=vector_space_model, weight=0.2,
                                n_actual_imposters=10, eps=0.01,
                                norm="l2", top_rank=3, balanced_test_pairs=False)
        logging.info("Starting verification [train / test]")
        verifier.fit(X_train, X_test)
        results, test_results = verifier.verify()

        logging.info("Computing results")
        dev_f, dev_p, dev_r, dev_t = evaluate(results)
        print np.nanmax(dev_f)
        best_t = dev_t[np.nanargmax(dev_f)]

        test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t) 
        f_scores.append(test_f)

    print vector_space_model, sum(f_scores), sum(f_scores) / len(f_scores)
    sb.plt.plot(feature_ranges, f_scores, label=vector_space_model)

sb.plt.legend(loc='best')

#plot_test_densities(results=results, dev_t=best_t)
#plot_test_results(test_fscores[:-1], test_thresholds, test_precisions[:-1], test_recalls[:-1], best_t)
