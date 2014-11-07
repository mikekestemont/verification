import numpy as np

cimport cython
cimport numpy as np

DTYPE = np.float64
ctypedef double DTYPE_t

cdef inline double float_max(double a, double b): return a if a >= b else b
cdef inline double float_min(double a, double b): return a if a <= b else b
cdef inline double square(double i): return i*i

@cython.boundscheck(False)
def min_max(np.ndarray[DTYPE_t, ndim=1] a_vec, np.ndarray[DTYPE_t, ndim=1] b_vec):
    assert a_vec.dtype == DTYPE and b_vec.dtype == DTYPE
    cdef double mins = 0.0
    cdef double maxs = 0.0
    cdef int i
    for i in range(a_vec.shape[0]):
        mins += float_min(a_vec[i], b_vec[i])
        maxs += float_max(a_vec[i], b_vec[i])
    if maxs > 0.0:
        return mins/maxs
    else:
        return 0.0

@cython.boundscheck(False)
def burrows_delta(np.ndarray[DTYPE_t, ndim=1] a_vec, np.ndarray[DTYPE_t, ndim=1] b_vec, np.ndarray[DTYPE_t, ndim=1] weights):
    assert a_vec.dtype == DTYPE and b_vec.dtype == DTYPE and weights.dtype == DTYPE
    cdef double delta = 0.0
    cdef int i
    for i in range(a_vec.shape[0]):
        delta += abs(a_vec[i]-b_vec[i])/weights[i]
    return delta

@cython.boundscheck(False)
def manhattan(np.ndarray[DTYPE_t, ndim=1] a_vec, np.ndarray[DTYPE_t, ndim=1] b_vec):
    assert a_vec.dtype == DTYPE and b_vec.dtype == DTYPE
    cdef double distance = 0.0
    cdef int i
    for i in range(a_vec.shape[0]):
        distance += abs(a_vec[i]-b_vec[i])
    return distance

@cython.boundscheck(False)
def ngd(np.ndarray[DTYPE_t, ndim=1] a_vec, np.ndarray[DTYPE_t, ndim=1] b_vec):
    # ngram-distance defined by Kesjl, Stamatatos etc.
    cdef double dist = 0.0
    cdef int i
    cdef double update = 0.0
    cdef double term_a = 0.0
    cdef double term_b = 0.0
    for i in range(a_vec.shape[0]):
        term_a = 2.0*(a_vec[i]-b_vec[i])
        term_b = a_vec[i]+b_vec[i]
        if term_b:
            update = square(term_a/term_b)
            if update:
                dist+=update
    return dist