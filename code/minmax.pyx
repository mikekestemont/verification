import numpy as np

cimport cython
cimport numpy as np

DTYPE = np.float64
ctypedef double DTYPE_t

cdef inline double float_max(double a, double b): return a if a >= b else b
cdef inline double float_min(double a, double b): return a if a <= b else b

@cython.boundscheck(False)
def min_max(np.ndarray[DTYPE_t, ndim=1] a_vec, np.ndarray[DTYPE_t, ndim=1] b_vec):
    assert a_vec.dtype == DTYPE and b_vec.dtype == DTYPE
    cdef double mins = 0.0
    cdef double maxs = 0.0
    cdef int i
    for i in range(a_vec.shape[0]):
        mins += float_min(a_vec[i], b_vec[i])
        maxs += float_max(a_vec[i], b_vec[i])
    return mins/maxs