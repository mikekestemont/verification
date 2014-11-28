import numpy as np

cimport cython
cimport numpy as np

cdef fused floating1d:
    float[::1]
    double[::1]

@cython.boundscheck(False)
def sparse_min_max(floating1d X_data, int[:] X_indices, int[:] X_indptr,
                   floating1d Y_data, int[:] Y_indices, int[:] Y_indptr,
                   int n_features, int[:] indices):
    cdef double mins = 0.0
    cdef double maxs = 0.0
    cdef double[::1] xrow = np.zeros(n_features)
    cdef double[::1] yrow = np.zeros(n_features)
    cdef np.npy_intp j, i
    for j in range(X_indptr[0], X_indptr[1]):
        xrow[X_indices[j]] = X_data[j]
    for j in range(Y_indptr[0], Y_indptr[1]):
        yrow[Y_indices[j]] = Y_data[j]
    for i in indices:
        a = xrow[i]
        b = yrow[i]
        if a >= b:
            maxs += a
            mins += b
        else:
            maxs += b
            mins += a
    if maxs > 0.0:
        return 1.0 - (mins / maxs)
    return 0.0

cdef inline double square(double i): return i * i

@cython.boundscheck(False)
def sparse_divergence(floating1d X_data, int[:] X_indices, int[:] X_indptr,
                      floating1d Y_data, int[:] Y_indices, int[:] Y_indptr,
                      int n_features):
    cdef double[::1] xrow = np.zeros(n_features)
    cdef double[::1] yrow = np.zeros(n_features)
    cdef np.npy_intp j, i
    cdef double divergence = 0.0
    for j in range(X_indptr[0], X_indptr[1]):
        xrow[X_indices[j]] = X_data[j]
    for j in range(Y_indptr[0], Y_indptr[1]):
        yrow[Y_indices[j]] = Y_data[j]
    for i in range(n_features):
        a, b = xrow[i], yrow[i]
        if a != b:
        divergence += square((2.0 * (a-b)) / (a+b))
    return divergence
