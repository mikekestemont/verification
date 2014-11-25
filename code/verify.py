from itertools import combinations

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from gensim.utils import tokenize
from plm import ParsimoniousLM

def analyzer(words, n):
    for word in words:
        if len(word) <= n:
            yield word
        else:
            word = "%" + word + "*"
            for i in range(len(word) - n - 1):
                yield word[i:i + n]


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
    'tf': Pipeline([('tf', TfidfVectorizer(analyzer=analyzer, use_idf=False))]),
    'std': Pipeline([('tf', TfidfVectorizer(analyzer=analyzer, use_idf=False)),
                     ('scaler', DeltaWeightScaler())]),
    'idf': Pipeline([('tf', CountVectorizer(analyzer=analyzer)),
                     ('tfidf', TfidfTransformer())]),
    'plm': Pipeline([('tf', ParsimoniousLM())])
}


class Verification(object):

    def __init__(self, n_features=1000, random_prop=0.5, sample_features=False,
                 sample_authors=False, metric='cosine', text_cutoff=None,
                 sample_iterations=100, n_potential_imposters=30,
                 n_actual_imposters=10, n_test_pairs=None, random_state=None,
                 vector_space_model='std', weight=0.1, em_iterations=100,
                 tfidf_norm='l2'):

        self.n_features = n_features
        self.random_prop = int(random_prop * n_features)
        self.sample_features = sample_features
        self.sample_authors = sample_authors
        self.metric = metric
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
            self.parameters['tf__weight'] = weight

    def fit(self, background_dataset, dev_dataset):
        self.train_data, self.train_titles, self.train_authors = background_dataset
        self.dev_data, self.dev_titles, self.dev_authors = dev_dataset
        transformer = pipelines[self.vector_space_model]
        transformer.set_params(self.parameters)
        transformer.fit(self.train_data + self.dev_data)
        self.X_train = transformer.transform(self.train_data)
        self.X_dev = transformer.transform(self.dev_data)
        return self

    def _setup_test_pairs(self):
        test_pairs = []
        for i, j in combinations(range(len(self.dev_titles)), 2):
            title_i, title_j = dev_titles[i], dev_titles[j]
            if title_i[:title_i.index("_")] != title_j[:title_j.index('_')]:
                test_pairs.append((i, j))
        self.rnd.shuffle(test_pairs)
        return test_pairs[:self.n_test_pairs]

    def _verification(self):
        distances, labels = [], []
        for i, j in self._setup_test_pairs():
            distances.append(self.metric(self.X_dev[i], self.X_dev[j]))
            labels.append(
                "same_author" if self.dev_authors[i] == self.dev_authors[j] else
                "diff_author")
        min_dist, max_dist = min(distances), max(distances)
        for distance, label in zip(distances, labels):
            yield label, (distance - min_dist) / (max_dist - min_dist)

    def _verification_with_sampling(self):
        for i, j in self._setup_test_pairs():
            author_i, author_j = self.dev_authors[i], self.dev_authors[j]
            train_sims = []
            for k in range(self.X_train[0]):
                if self.train_authors[k] not in (author_i, author_j):
                    train_sims.append(k, self.train_authors[k],
                                      self.metric(self.X_dev[i], self.X_train[k]))
            train_sims.sort(key=lambda sim: s[-1])
            indexes, imposters, _ = zip(*train_sim[:self.n_potential_imposters])
            X_imposters = self.X_train[list(imposters)]
            for iteration in range(self.sample_iterations):
                rnd_imposters = self.rnd.randint(
                    X_imposters.shape[0], size=self.n_actual_imposters)
                X_truncated = X_imposters[rnd_imposters, :]
                rnd_features = self.rnd.randint(
                    X_truncated.shape[1], size=self.random_prop)
                X_truncated = X_truncated[:, rnd_features]


    def verify(self):
        if self.sample_authors or self.sample_features:
            return self._verification_with_sampling()
        return self._verification()
