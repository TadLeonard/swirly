import numpy as np
from numpy import ndarray


def chunk_select(ndarray indices):
    """Generate contiguous chunks of indices in tuples of
    (start_index, stop_index) where stop_index is not inclusive"""
    cdef ndarray contiguous = np.diff(indices) == 1 
    cdef int longest = max(contiguous.size, indices.size)
    cdef int i = 0
    cdef list out = []
    cdef int left, right
    while i < longest:
        left = indices[i]
        if not contiguous[i]:
            out.append((left, left + 1))
            yield left, left + 1
        else:
            while i < longest:
                right = indices[i]
                if not contiguous[i]:
                    out.append((left, left + 1))
                    break
    return out
             

