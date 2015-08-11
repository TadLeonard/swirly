import numpy as np
cimport numpy as np

from cpython cimport bool
import cython
from cython.parallel import parallel, prange


ctypedef np.int_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def chunk_select(np.ndarray[np.int_t, ndim=1] indices):
    """Generate contiguous chunks of indices in tuples of
    (start_index, stop_index) where stop_index is not inclusive"""
    cdef np.ndarray contiguous = np.diff(indices) == 1 
    cdef int shortest = contiguous.size
    cdef int longest = indices.size
    cdef int left, right, i
    cdef list out = []
    i = -1
    while i < (longest - 1):
        i += 1
        left = indices[i]
        if i >= shortest or not contiguous[i]:
            out.append((left, left + 1))
        else:
            while i < (longest - 1):
                i += 1
                right = indices[i]
                if i >= shortest or not contiguous[i]:
                    out.append((left, right + 1))
                    break
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def avg_col_height(
        np.ndarray[DTYPE_t, ndim=1] col_indices,
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
        

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def nonzero2d(np.ndarray[np.uint8_t, ndim=2] select):
    #cdef list rows = []
    #cdef list cols = []
    cdef list avgs = []  # avg col idx per row idx
    cdef list rowset = []  # set of non empty rows
    cdef int total_avg   # total avg col idx
    cdef int total_sum = 0
    cdef int total_count = 0
    cdef int i, j
    cdef int current_sum = 0
    cdef int current_winsize = 0
    
    for i in range(select.shape[0]):
        current_sum = current_winsize = 0
        for j in range(select.shape[1]):
            if select[i, j]:
                #rows.append(i)
                #cols.append(j)
                current_sum += j
                current_winsize += 1
        if current_winsize != 0:
            total_count += current_winsize
            total_sum += current_sum
            avgs.append(current_sum / current_winsize)
            rowset.append(i)
    total_avg = total_sum / total_count 
    return avgs, rowset, total_avg
    

