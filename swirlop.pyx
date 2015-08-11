import numpy as np
cimport numpy as np

from cpython cimport bool
import cython
from cython.parallel import parallel, prange


ctypedef np.int_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def chunk_select(np.ndarray indices):
    """Generate contiguous chunks of indices in tuples of
    (start_index, stop_index) where stop_index is not inclusive"""
    assert indices.dtype == int

    cdef np.ndarray contiguous = np.diff(indices) == 1 
    cdef int shortest = contiguous.size
    cdef int longest = indices.size
    cdef int left, right, i
    i = -1
    while i < (longest - 1):
        i += 1
        left = indices[i]
        if i >= shortest or not contiguous[i]:
            yield left, left + 1
        else:
            while i < (longest - 1):
                i += 1
                right = indices[i]
                if i >= shortest or not contiguous[i]:
                    yield left, right + 1
                    break



@cython.boundscheck(False)
@cython.wraparound(False)
def avg_col_height(np.ndarray[DTYPE_t, ndim=1] col_indices,
                   np.ndarray[DTYPE_t, ndim=1] row_indices,
                   np.ndarray[DTYPE_t, ndim=1] cols):
    assert col_indices.dtype == int
    assert row_indices.dtype == int
    assert cols.dtype == int

    cdef np.ndarray[DTYPE_t, ndim=1] avgs
    avgs = np.empty(cols.shape[0], dtype=np.int)
    cdef int i
    cdef int current_col = col_indices[0]
    cdef int current_idx = 0
    cdef int current_sum = 0
    cdef int current_winsize = 0
    cdef int current_row = 0
    for i in range(col_indices.shape[0]):
        if current_col != col_indices[i]:
            current_col = col_indices[i]
            avgs[current_idx] = current_sum / current_winsize
            current_sum = 0
            current_winsize = 0
            current_idx += 1
        current_row = row_indices[i]
        current_sum += current_row
        current_winsize += 1
    avgs[current_idx] = current_sum / current_winsize  # final avg
    return avgs 
        
