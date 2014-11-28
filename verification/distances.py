from sparse_distances import sparse_min_max, sparse_divergence, sparse_cosine
from sparse_distances import sparse_euclidean, sparse_cityblock
import numpy as np

def _dist(X, Y, fn, indices):
    if indices is None:
        indices = list(set(X.indices.tolist() + Y.indices.tolist()))
    indices = np.array(indices, dtype=np.int32)
    return fn(X.data, X.indices, X.indptr,
              Y.data, Y.indices, Y.indptr,
              X.shape[1], indices)

def minmax(X, Y, indices=None):
    return _dist(X, Y, sparse_min_max, indices)

def divergence(X, Y, indices=None):
    return _dist(X, Y, sparse_divergence, indices)

def cityblock(X, Y, indices=None):
    return _dist(X, Y, sparse_cityblock, indices)

def euclidean(X, Y, indices=None):
    return _dist(X, Y, sparse_euclidean, indices)

def cosine(X, Y, indices=None):
    return _dist(X, Y, sparse_cosine, indices)
