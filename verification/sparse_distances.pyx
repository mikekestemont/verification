import numpy as np
cimport cython
cimport numpy as np
from libc.stdlib cimport *
from libc.math cimport fabs, sqrt

from scipy.linalg.blas import fblas


cdef fused floating1d:
    float[::1]
    double[::1]

@cython.boundscheck(False)
cdef double norm(double[:] x, int[:] indices):
    cdef double ans = 0.0
    cdef double _d
    cdef size_t i
    for i in range(indices.shape[0]):
        _d = x[indices[i]] * x[indices[i]]
        assert not (np.isnan(_d) or np.isinf(_d))
        ans += _d
    return sqrt(ans)

@cython.boundscheck(False)
def sparse_euclidean(floating1d X_data, int[:] X_indices, int[:] X_indptr,
                     floating1d Y_data, int[:] Y_indices, int[:] Y_indptr,
                     int n_features, int[:] indices):
    cdef double[::1] row = np.zeros(n_features)
    cdef double dist = 0.0
    cdef np.npy_intp j
    for j in range(X_indptr[0], X_indptr[1]):
        row[X_indices[j]] = X_data[j]
    for j in range(Y_indptr[0], Y_indptr[1]):
        row[Y_indices[j]] -= Y_data[j]
    return norm(row, indices)

@cython.boundscheck(False)
def sparse_cityblock(floating1d X_data, int[:] X_indices, int[:] X_indptr,
                     floating1d Y_data, int[:] Y_indices, int[:] Y_indptr,
                     int n_features, int[:] indices):
    cdef double[::1] row = np.zeros(n_features)
    cdef double dist = 0.0
    cdef np.npy_intp j
    for j in range(X_indptr[0], X_indptr[1]):
        row[X_indices[j]] = X_data[j]
    for j in range(Y_indptr[0], Y_indptr[1]):
        row[Y_indices[j]] = fabs(row[Y_indices[j]] - Y_data[j])
    dist = fblas.dasum(row)
    return dist

@cython.boundscheck(False)
cdef double dot_product(double[:] x, double[:] y, int[:] indices):
    cdef double s = 0.0
    cdef size_t i
    for i in range(indices.shape[0]):
        s += x[indices[i]] * y[indices[i]]
    return s

@cython.cdivision(True)
@cython.boundscheck(False)
def sparse_cosine(floating1d X_data, int[:] X_indices, int[:] X_indptr,
                  floating1d Y_data, int[:] Y_indices, int[:] Y_indptr,
                  int n_features, int[:] indices):
    cdef double[::1] xrow = np.zeros(n_features)
    cdef double[::1] yrow = np.zeros(n_features)
    cdef np.npy_intp j
    for j in range(X_indptr[0], X_indptr[1]):
        xrow[X_indices[j]] = X_data[j]
    for j in range(Y_indptr[0], Y_indptr[1]):
        yrow[Y_indices[j]] = Y_data[j]
    return (1 - (dot_product(xrow, yrow, indices) /
                (norm(xrow, indices) * norm(yrow, indices))))

@cython.cdivision(True)
@cython.boundscheck(False)
def sparse_min_max(floating1d X_data, int[:] X_indices, int[:] X_indptr,
                   floating1d Y_data, int[:] Y_indices, int[:] Y_indptr,
                   int n_features, int[:] indices):
    cdef double mins = 0.0
    cdef double maxs = 0.0
    cdef double a, b
    cdef double[::1] xrow = np.zeros(n_features, dtype=np.float64)
    cdef double[::1] yrow = np.zeros(n_features, dtype=np.float64)
    cdef np.npy_intp j
    cdef size_t i
    for j in range(X_indptr[0], X_indptr[1]):
        xrow[X_indices[j]] = X_data[j]
    for j in range(Y_indptr[0], Y_indptr[1]):
        yrow[Y_indices[j]] = Y_data[j]
    for i in range(indices.shape[0]):
        a = xrow[indices[i]]
        b = yrow[indices[i]]
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

@cython.cdivision(True)
@cython.boundscheck(False)
def sparse_divergence(floating1d X_data, int[:] X_indices, int[:] X_indptr,
                      floating1d Y_data, int[:] Y_indices, int[:] Y_indptr,
                      int n_features, int[:] indices):
    cdef double[::1] xrow = np.zeros(n_features, dtype=np.float64)
    cdef double[::1] yrow = np.zeros(n_features, dtype=np.float64)
    cdef np.npy_intp j
    cdef size_t i
    cdef double divergence = 0.0
    cdef double a, b
    for j in range(X_indptr[0], X_indptr[1]):
        xrow[X_indices[j]] = X_data[j]
    for j in range(Y_indptr[0], Y_indptr[1]):
        yrow[Y_indices[j]] = Y_data[j]
    for i in range(indices.shape[0]):
        a = xrow[indices[i]]
        b = yrow[indices[i]]
        if a != b:
            divergence += square((2.0 * (a-b)) / (a+b))
    return divergence
