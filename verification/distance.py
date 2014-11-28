from distances_sparse import sparse_min_max
import numpy as np

def minmax(X, Y):
    indices = np.array(list(set(X.indices.tolist() + Y.indices.tolist())), dtype=np.int32)
    return sparse_min_max(X.data, X.indices, X.indptr,
                          Y.data, Y.indices, Y.indptr,
                          X.shape[1], indices)
