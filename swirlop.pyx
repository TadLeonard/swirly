import numpy as np
cimport numpy as np

from cpython cimport bool
import cython


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
def column_avgs(np.ndarray[np.uint8_t, ndim=2] select):
    """Computes the average column index for each row
    in a 2D boolean mask array. Useful byproducts are the
    total column index average and the unique set of rows involved"""
    cdef list avgs = []  # avg col idx per row idx
    cdef list rowset = []  # set of non empty rows
    cdef long total_avg   # total avg col idx
    cdef long total_sum = 0
    cdef int total_count = 0
    cdef int i, j
    cdef long current_sum = 0
    cdef int current_winsize = 0
    
    for i in range(select.shape[0]):
        current_sum = current_winsize = 0
        for j in range(select.shape[1]):
            if select[i, j]:
                current_sum += j
                current_winsize += 1
        if current_winsize != 0:
            total_count += current_winsize
            total_sum += current_sum
            avgs.append(current_sum / current_winsize)
            rowset.append(i)
    total_avg = total_sum / total_count 
    return avgs, rowset, total_avg
    

