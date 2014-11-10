import numpy as np

cimport cython
cimport numpy as np

DTYPE = np.float64
ctypedef double DTYPE_t

cdef inline double float_max(double a, double b): return a if a >= b else b
cdef inline double float_min(double a, double b): return a if a <= b else b
cdef inline double square(double i): return i*i

@cython.boundscheck(False)
def minmax(np.ndarray[DTYPE_t, ndim=1] a_vec, np.ndarray[DTYPE_t, ndim=1] b_vec):
    assert a_vec.dtype == DTYPE and b_vec.dtype == DTYPE
    cdef double mins = 0.0
    cdef double maxs = 0.0
    cdef int i
    for i in range(a_vec.shape[0]):
        mins += float_min(a_vec[i], b_vec[i])
        maxs += float_max(a_vec[i], b_vec[i])
    if maxs > 0.0:
        return 1.0-(mins/maxs)
    else:
        return 0.0

@cython.boundscheck(False)
def divergence(np.ndarray[DTYPE_t, ndim=1] a_vec, np.ndarray[DTYPE_t, ndim=1] b_vec):
    # ngram-distance defined by Keselj, Stamatatos etc.
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