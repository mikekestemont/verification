from distances_sparse import sparse_min_max, sparse_divergence
import numpy as np

def _dist(X, Y, fn):
    indices = np.array(list(set(X.indices.tolist() + Y.indices.tolist())), dtype=np.int32)
    return fn(X.data, X.indices, X.indptr,
              Y.data, Y.indices, Y.indptr,
              X.shape[1], indices)

def minmax(X, Y):
    return _dist(X, Y, sparse_min_max)

def divergence(X, Y):
    return _dist(X, Y, sparse_divergence)
