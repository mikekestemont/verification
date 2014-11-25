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
                 n_actual_imposters=10, n_test_pairs=1000, random_state=None,
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
        transformer = pipelines[self.vector_space_model]
        transformer.set_params(self.parameters)
        transformer.fit(background_dataset + dev_dataset)
        self.X_train = transformer.transform(background_dataset)
        self.X_dev = transformer.transform(dev_dataset)
        return self
