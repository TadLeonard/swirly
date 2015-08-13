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
    cdef int nrows = select.shape[0]
    cdef int ncols = select.shape[1] 
    cdef np.ndarray[np.float64_t, ndim=1] avgs
    avgs = np.empty((nrows,), dtype=np.float64)
    avgs[:] = -1.0
    cdef float total_avg   # total avg col idx
    cdef float total_sum = 0
    cdef int total_count = 0
    cdef int i, j
    cdef float current_sum = 0
    cdef int current_winsize = 0
    cdef unsigned int value
    
    for i in range(nrows):
        current_sum = current_winsize = 0
        for j in range(ncols):
            value = <unsigned int>(select[i, j]) 
            current_sum += (j * value)
            current_winsize += (1 * value)
        total_count += current_winsize
        total_sum += current_sum
        if current_winsize != 0:
            avgs[i] = current_sum / current_winsize
    total_avg = total_sum / total_count 
    return avgs, total_avg
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def move(
        np.ndarray[np.uint8_t, ndim=3] img,
        np.ndarray[np.uint8_t, ndim=2] select,
        int travel):
    """Moves slices of an image's columns in a certain direction
    by `select` steps. Moves the 2D boolean selection mask along with it."""
    if travel < 0:
        img = img[::-1]
        select = select[::-1]
        travel = -travel
    move3d(img, travel)
    move2d(select, travel)


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
cpdef inline void move2d_many(
        np.ndarray[np.uint8_t, ndim=3] img,
        np.ndarray[np.int_t, ndim=1] cols_to_move,
        int travel):
    tails = np.empty((cols_to_move, travel), dtype=np.uint8)
   # for i in range(cols_to_move.shape[0]):
   #     tails[i] = make_tail_1d(
        
    return

 
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


#cpdef void move_3d_row(np.ndarray[np.uint8_t, ndim=2] img, int travel):
#    cdef int i, c
#    def cols = img.shape[0]
#    def chans = img.shape[1]
#    cpdef np.ndarray[np.uint8_t, ndim=2] tail
#    tail = np.empty((travel, rows), dtype=np.uint8)
#    for i in range(rows):
#        for i in range(travel):
#            tail[


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef inline void move1d(np.ndarray[np.uint8_t, ndim=1] img, int travel):
    cdef int i
    cdef int cols = img.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] tail
    tail = np.empty((travel,), dtype=np.uint8)
    for i in range(travel):
        tail[i] = img[i + cols - travel]
    for i in range(cols-1, travel-1, -1):
        img[i] = img[i - travel]
    for i in range(travel):
        img[i] = tail[i]

 
