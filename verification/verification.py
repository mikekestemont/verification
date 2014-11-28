import logging
from functools import partial
from itertools import combinations
from operator import itemgetter

import numpy as np
import scipy.sparse as sp
import seaborn as sb

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import dist_metrics
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from gensim.utils import tokenize
from sparse_plm import SparsePLM

from preprocessing import analyzer, identity
from distances import minmax, divergence, cityblock, cosine, euclidean

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

rc = {'axes.labelsize': 3, 'font.size': 3, 'legend.fontsize': 3.0,
      'axes.titlesize': 3, "font.family": "sans-serif",
      'xlabel.major.size': 0.3, 'xlabel.minor.size': 0.3,
      'ylabel.major.size': 0.3, 'ylabel.minor.size': 0.3,
      'font.family': 'Arial', 'font.sans-serif': ['Bitstream Vera Sans']}
sb.set_style("darkgrid", rc=rc)


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
    "manhattan": cityblock,
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
                 tfidf_norm='l2'):

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
        self.parameters = {'tf__max_features': n_features}
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
            logging.info("Verifying pair %s / %s" % (k+1, len(test_pairs)+1))
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
            logging.info("Verifying pair %s / %s" % (k+1, len(test_pairs)+1))
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
        if self.sample_authors or self.sample_features:
            return self._verification_with_sampling()
        return self._verification()

def get_result_for_threshold(results, t, sample=False):
    preds, true = [], []
    # kan als we 1 - sigmas.mean() gebruiken, weg
    for label, score in results:
        if sample:
            preds.append(1 if score >= t else 0)
        else:
            preds.append(1 if score <= t else 0)
        true.append(1 if label == "same_author" else 0)
    return (f1_score(true, preds), t,
            precision_score(true, preds), recall_score(true, preds))

def evaluate_predictions(results, sample=False):
    results = list(results)
    dev_results = results[:int(len(results) / 2.0)]
    test_results = results[int(len(results) / 2.0):]
    thresholds = np.arange(0.001, 1.001, 0.001)
    dev_f1, dev_t, _, _ = max(
        (get_result_for_threshold(dev_results, t, sample) for t in thresholds),
        key=itemgetter(0))
    test_scores = [get_result_for_threshold(test_results, t, sample)
                   for t in thresholds]
    return test_scores, dev_t

def prec_recall_curve(scores, dev_t, filename="prec_rec.pdf", fontsize=7):
    fig = sb.plt.figure()
    sb.plt.xlabel("recall", fontsize=fontsize)
    sb.plt.ylabel("precision", fontsize=fontsize)
    sb.plt.xlim(0, 1); sb.plt.ylim(0, 1)
    _, _, precisions, recalls = zip(*scores)
    sb.plt.plot(precisions, recalls)
    sb.plt.savefig(filename)

def plot_test_densities(results, dev_t, filename="test_densities.pdf",
                        fontsize=7):
    fig = sb.plt.figure()
    same_author_densities = np.asarray(
        [sc for c, sc in results if c == "same_author"])
    diff_author_densities = np.asarray(
        [sc for c, sc in results if c == "diff_author"])
    c1, c2, c3 = sb.color_palette("Set1")[:3]
    sb.plt.xlim(0, 1)
    sb.kdeplot(diff_author_densities, shade=True,
               label="Different author pairs", legend=False, c=c1)
    sb.kdeplot(same_author_densities, shade=True,
               label="Same author pairs", legend=False, c=c2)
    sb.plt.axvline(x=dev_t, linewidth=1, c=c3) # good idea?
    sb.plt.legend(loc=0)
    sb.plt.savefig(filename)
    sb.plt.clf()

def plot_test_results(scores, dev_t, filename="test_curve.pdf", fontsize=7):
    fig = sb.plt.figure()
    c1, c2, c3, c4 = sb.color_palette("Set1")[:4]
    f1_scores, thresholds, precisions, recalls = zip(*scores)
    sb.plt.plot(f1_scores, thresholds, label="F1 score", c=c1)
    sb.plt.plot(precisions, thresholds, label="Precision", c=c2)
    sb.plt.plot(recalls, thresholds, label="Recall", c=c3)
    sb.plt.xlim(0, 1)
    sb.plt.ylim(0, 1.005)
    sb.plt.axvline(x=dev_t, linewidth=1, c=c4)
    sb.plt.legend(loc=0)
    sb.plt.xlabel('Threshold', fontsize=fontsize)
    sb.plt.ylabel('Score', fontsize=fontsize)
    sb.plt.savefig(filename)
    sb.plt.clf()
