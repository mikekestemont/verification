import logging
from functools import partial
from itertools import combinations
from operator import itemgetter

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import dist_metrics
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from gensim.utils import tokenize
from sparse_plm import SparsePLM

from preprocessing import analyzer, identity
from distances import minmax, divergence, cityblock, cosine, euclidean
from plotting import prec_recall_curve, plot_test_densities, plot_test_results


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class DeltaWeightScaler(BaseEstimator):

    def fit(self, X, y=None):
        self.weights = StandardScaler(with_mean=False).fit(X).std_
        return self

    def transform(self, X):
        X = sp.csr_matrix(X, dtype=np.float64)
        for i in range(X.shape[0]):
            start, end = X.indptr[i], X.indptr[i+1]
            X.data[start:end] /= self.weights[X.indices[start:end]]
        return X


pipelines = {
    'tf': Pipeline([('tf', TfidfVectorizer(analyzer=identity, use_idf=False))]),
    'std': Pipeline([('tf', TfidfVectorizer(analyzer=identity, use_idf=False)),
                     ('scaler', DeltaWeightScaler())]),
    'idf': Pipeline([('tf', CountVectorizer(analyzer=identity)),
                     ('tfidf', TfidfTransformer())]),
    'plm': Pipeline([('tf', CountVectorizer(analyzer=identity)),
                     ('plm', SparsePLM())])
}

distance_metrics = {
    "minmax": minmax,
    "cityblock": cityblock,
    "euclidean": euclidean,
    "cosine": cosine,
    "divergence": divergence,
}

class Verification(object):

    def __init__(self, n_features=1000, random_prop=0.5, sample_features=False,
                 sample_authors=False, metric='cosine', text_cutoff=None,
                 sample_iterations=10, n_potential_imposters=30,
                 n_actual_imposters=10, n_test_pairs=None, random_state=None,
                 vector_space_model='std', weight=0.1, em_iterations=10,
                 ngram_range=(1, 1), tfidf_norm='l2'):

        self.n_features = n_features
        self.random_prop = int(random_prop * n_features)
        self.sample_features = sample_features
        self.sample_authors = sample_authors
        self.metric = distance_metrics[metric]
        self.text_cutoff = text_cutoff
        self.sample_iterations = sample_iterations
        self.n_potential_imposters = n_potential_imposters
        self.n_actual_imposters = n_actual_imposters
        self.n_test_pairs = n_test_pairs
        self.vector_space_model = vector_space_model
        self.weight = weight
        self.em_iterations = em_iterations
        self.tfidf_norm = tfidf_norm
        self.rnd = np.random.RandomState(random_state)
        self.parameters = {'tf__max_features': n_features,
                           'tf__ngram_range': ngram_range}
        if self.vector_space_model == 'idf':
            self.parameters['tfidf__norm'] = tfidf_norm
        elif self.vector_space_model == 'plm':
            self.parameters['plm__weight'] = weight
            self.parameters['plm__iterations'] = em_iterations

    def fit(self, background_dataset, dev_dataset):
        self.train_data, self.train_titles, self.train_authors = background_dataset
        self.dev_data, self.dev_titles, self.dev_authors = dev_dataset
        transformer = pipelines[self.vector_space_model]
        transformer.set_params(**self.parameters)
        transformer.fit(self.train_data + self.dev_data)
        self.X_train = transformer.transform(self.train_data)
        logging.info("Background corpus: n_samples=%s / n_features=%s" % (
            self.X_train.shape))
        self.X_dev = transformer.transform(self.dev_data)
        logging.info("Development corpus: n_samples=%s / n_features=%s" % (
            self.X_train.shape))
        return self

    def _setup_test_pairs(self):
        test_pairs = []
        for i in range(len(self.dev_titles)):
            for j in range(len(self.dev_titles)):
                if i != j:
                    title_i, title_j = self.dev_titles[i], self.dev_titles[j]
                    if title_i.split("_")[0] != title_j.split('_')[0]:
                        test_pairs.append((i, j))
        self.rnd.shuffle(test_pairs)
        return test_pairs[:self.n_test_pairs]

    def _verification(self):
        distances, labels = [], []
        test_pairs = self._setup_test_pairs()
        for k, (i, j) in enumerate(test_pairs):
            logging.info("Verifying pair %s / %s" % (k+1, len(test_pairs)))
            distances.append(self.metric(self.X_dev[i], self.X_dev[j]))
            labels.append(
                "same_author" if self.dev_authors[i] == self.dev_authors[j] else
                "diff_author")
        min_dist, max_dist = min(distances), max(distances)
        for distance, label in zip(distances, labels):
            yield label, (distance - min_dist) / (max_dist - min_dist)

    def _verification_with_sampling(self):
        test_pairs = self._setup_test_pairs()
        for k, (i, j) in enumerate(test_pairs):
            logging.info("Verifying pair %s / %s" % (k+1, len(test_pairs)))
            author_i, author_j = self.dev_authors[i], self.dev_authors[j]
            train_sims = []
            for k in range(self.X_train.shape[0]):
                if self.train_authors[k] not in (author_i, author_j):
                    train_sims.append((k, self.train_authors[k],
                                       self.metric(self.X_dev[i], self.X_train[k])))
            train_sims.sort(key=lambda sim: sim[-1])
            indexes, imposters, _ = zip(*train_sims[:self.n_potential_imposters])
            X_imposters = self.X_train[list(indexes), :]
            targets = 0.0
            sigmas = np.zeros(self.sample_iterations)
            for iteration in range(self.sample_iterations):
                rnd_imposters = self.rnd.randint(
                    X_imposters.shape[0], size=self.n_actual_imposters)
                X_truncated = X_imposters[rnd_imposters, :]
                rnd_features = self.rnd.randint(
                    X_truncated.shape[1], size=self.random_prop)
                vec_i, vec_j = self.X_dev[i], self.X_dev[j]
                most_similar = min(self.metric(vec_i, X_truncated[k], rnd_features)
                                   for k in range(self.n_actual_imposters))
                target_dist = self.metric(vec_i, vec_j, rnd_features)
                if target_dist <= most_similar:
                    targets += 1
                sigmas[iteration] = targets / (iteration + 1.0)
            yield ("same_author" if author_i == author_j else "diff_author",
                   1 - sigmas.mean())

    def verify(self):
        "Start the verification procedure."
        if self.sample_authors or self.sample_features:
            return self._verification_with_sampling()
        return self._verification()


def evaluate(results, beta=2):
    y_true = np.array([0 if l == "diff_author" else 1 for l, _ in results])
    scores = np.array([score for _, score in results])
    scores = 1 - scores # highest similarity highest in rank
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f_scores = beta * (precisions * recalls) / (precisions + recalls)
    return f_scores, precisions, recalls, thresholds

def evaluate_with_threshold(results, t, beta=2):
    y_true = np.array([0 if l == "diff_author" else 1 for l, _ in results])
    preds = np.array([1 if score <= t else 0 for _, score in results])
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f_score = beta * (precision * recall) / (precision + recall)
    return f_score, precision, recall


# def evaluate_predictions(results, sample=False):
#     """Given a list of tuples of (label, score), return all scores
#     for 50% of the tuples at the best threshold in the other 50%."""
#     results = list(results)
#     dev_results = results[:int(len(results) / 2.0)]
#     test_results = results[int(len(results) / 2.0):]
#     thresholds = np.arange(0.001, 1.001, 0.001)
#     dev_f1, dev_t, _, _ = max(
#         (get_result_for_threshold(dev_results, t, sample) for t in thresholds),
#         key=itemgetter(0))
#     Test_scores = [get_result_for_threshold(test_results, t, sample)
#                    for t in thresholds]
#     return test_scores, dev_t
