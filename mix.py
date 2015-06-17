from __future__ import print_function
import sys
import time
import imread
import numpy as np


def timeit(fn):
    def new_fn(*args, **kwargs):
        start = time.time()
        ret = fn(*args, **kwargs)
        print("{} {:0.4f}s".format(fn.__name__, time.time() - start))
        return ret
    return new_fn


@timeit
def make_grid(img):
    rows = np.sum(img, axis=1)
    cols = np.sum(img, axis=0)
    print(img.shape)
    print(rows)
    print('--')
    print(rows.shape)
    print('--')
    print(cols.shape)


if __name__ == "__main__":
    img = imread.imread(sys.argv[1])
    make_grid(img)
