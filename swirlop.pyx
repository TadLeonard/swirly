import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange


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
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void movend(np.ndarray img, int travel):
    if travel < 0:
        img = img[::-1]
        travel = -travel

    cdef int nd = img.ndim
    if nd == 3:
        move3d(img, travel)
    elif nd == 2:
        move2d(img, travel)
    else:
        move1d(img, travel)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef inline void move3d(np.ndarray[np.uint8_t, ndim=3] img, int travel):
    cdef int i, j, c, x
    cdef int cols = img.shape[0]
    cdef int rows = img.shape[1]
    cdef int chans = img.shape[2]
    cdef np.ndarray[np.uint8_t, ndim=3] tail
    tail = np.empty((travel, rows, chans), dtype=np.uint8)
    for i in range(rows):
        for j in range(travel):  # cuts into cols
            x = j + cols - travel
            for c in range(chans):
                tail[j, i, c] = img[x, i, c]
    for i in range(rows):
        for j in range(cols-1, travel-1, -1):
            x = j - travel
            for c in range(chans):
                img[j, i, c] = img[x, i, c]
    for i in range(rows):
        for j in range(travel):
            for c in range(chans):
                img[j, i, c] = tail[j, i, c]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef inline void move2d(np.ndarray[np.uint8_t, ndim=2] img, int travel):
    cdef int i, j, x 
    cdef int cols = img.shape[0]
    cdef int rows = img.shape[1]
    cdef np.ndarray[np.uint8_t, ndim=2] tail
    tail = np.empty((travel, rows), dtype=np.uint8)
    for i in range(rows):
        for j in range(travel):
            x = j + cols - travel
            tail[j, i] = img[x, i]
    for i in range(rows):
        for j in range(cols-1, travel-1, -1):
            x = j - travel
            img[j, i] = img[x, i]
    for i in range(rows):
        for j in range(travel):
            img[j, i] = tail[j, i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef inline void move1d(np.ndarray[np.uint8_t, ndim=1] img, int travel):
    cdef int i, x
    cdef int cols = img.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] tail
    tail = np.empty((travel,), dtype=np.uint8)
    for i in range(travel):
        x = i + cols - travel
        tail[i] = img[x]
    for i in range(cols-1, travel-1, -1):
        x = i - travel
        img[i] = img[x]
    for i in range(travel):
        img[i] = tail[i]

 
