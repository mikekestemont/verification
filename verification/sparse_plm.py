import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator


class SparsePLM(BaseEstimator):
    def __init__(self, weight=0.1, iterations=50, eps=1e-9):
        self.weight = weight
        self.iterations = iterations
        self.eps = eps

    def fit(self, X, y=None):
        cf = np.array(X.sum(axis=0), dtype=np.float64)[0]
        self.pc = cf / np.sum(cf) * (1 - self.weight)
        return self

    def transform(self, X, copy=True):
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)
        for i in range(X.shape[0]):
            begin_col, end_col = X.indptr[i], X.indptr[i+1]
            data = X.data[begin_col: end_col]
            p_data = np.ones(data.shape[0]) / data.shape[0]
            c_data = self.pc[X.indices[begin_col: end_col]]
            for iteration in range(1, self.iterations + 1):
                p_data *= self.weight
                E = data * p_data / (c_data + p_data)
                M = E / E.sum()
                diff = np.abs(M - p_data)
                p_data = M
                if (diff < self.eps).all():
                    break
            _d = np.dot(p_data, p_data)
            assert not (np.isnan(_d) or np.isinf(_d))
            X.data[begin_col: end_col] = p_data
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
