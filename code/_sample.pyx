import numpy as np
cimport numpy as np
cimport cython

np.import_array()

ctypedef np.double_t DTYPE_t

cdef inline DTYPE_t double_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b
cdef inline DTYPE_t double_min(DTYPE_t a, DTYPE_t b): return a if a <= b else b


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline DTYPE_t min_max(np.ndarray[DTYPE_t, ndim=1] a, np.ndarray[DTYPE_t, ndim=1] b):
    cdef DTYPE_t mins = 0.0, maxs = 0.0, value = 0.0
    cdef int N = a.shape[0]
    cdef unsigned int i
    for i in range(N):
        mins += double_min(a[i], b[i])
        maxs += double_max(a[i], b[i])
    value = mins / maxs
    return value


def sample(int n_samples, int n_features, int n_imposters,
           np.ndarray[DTYPE_t, ndim=2] X, 
           np.ndarray[DTYPE_t, ndim=1] vec_i,
           np.ndarray[DTYPE_t, ndim=1] vec_j):
    cdef np.ndarray[np.int_t, ndim=1] indices
    cdef np.ndarray[DTYPE_t, ndim=2] truncated_X
    cdef np.ndarray[DTYPE_t, ndim=1] vec_i_trunc
    cdef int value = 0
    cdef DTYPE_t most_similar = -1, score
    cdef unsigned int idx
    indices = np.random.randint(0, n_samples, size=n_features)
    truncated_X = X[:,indices]
    vec_i_trunc = vec_i[indices]
    for idx in range(n_imposters):
        score = min_max(truncated_X[idx], vec_i_trunc)
        if score > most_similar:
            most_similar = score;
    if min_max(vec_j[indices], vec_i_trunc) >= most_similar:
        value = 1
    return value




