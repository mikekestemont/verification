import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.WARN)

import os
from verification.verification import Verification
from verification.evaluation import evaluate, evaluate_with_threshold, average_precision_score
from verification.evaluation import rank_predict
from verification.plotting import plot_test_densities, plot_test_results
from verification.preprocessing import prepare_corpus, Dataset
from sklearn.cross_validation import train_test_split
import numpy as np
import seaborn as sb

folders = os.listdir('../data')
scores = []
for folder in folders:
    folder = os.path.join('../data', folder)
    if 'caesar' not in folder and '1001' not in folder and os.path.isdir(folder):
        train, test = folder, folder
        logging.info("preparing corpus")
        print train, test
        if train == test:
            data = prepare_corpus(train)
            train_texts, test_texts, train_titles, test_titles, train_authors, test_authors = train_test_split(
                data.texts, data.titles, data.authors, test_size=0.5, random_state=1956)
            X_train = Dataset(train_texts.tolist(), train_titles.tolist(), train_authors.tolist())
            X_test = Dataset(test_texts.tolist(), test_titles.tolist(), test_authors.tolist())
        else:
            X_train = prepare_corpus(train)
            X_test = prepare_corpus(test)
        V = len(X_train.texts) + len(X_test.texts)
        Vn = len(set(sum(X_train.texts, []) + sum(X_test.texts, [])))
        vector_space_model = "plm"
        verifier = Verification(random_state=1,
                                metric='minmax', sample_authors=False,
                                n_features=Vn,
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
        test_fscores, test_precisions, test_recalls, test_thresholds = evaluate(test_results)

        sb.plt.plot(test_recalls, test_precisions, label=V)
        print "F: %.3f, P: %.3f, R: %.3f, AP: %.3f" % (
            test_f, test_p, test_r, average_precision_score(test_results))

        N = sum(l == "same_author" for l, _ in results)
        print N
        print rank_predict(test_results, method="proportional", N=N)
        print rank_predict(test_results, method="calibrated")
        print rank_predict(test_results, method="break-even-point")
        sb.plt.legend(loc='best')
        scores.append((V, test_f))

#plot_test_densities(results=results, dev_t=best_t)
#plot_test_results(test_fscores[:-1], test_thresholds, test_precisions[:-1], test_recalls[:-1], best_t)
