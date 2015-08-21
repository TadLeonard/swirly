"""Functions for moving pixels. These functions carry out the
mutations dictated by effect generators."""

from functools import partial

import numpy as np
from _swirlop import chunk_select, column_avgs, move_rubix, move_swap


###################
# Filtering pixels

_or = np.logical_or
_and = np.logical_and


def _choose(chooser, starter, img, *selections):
    sel = starter(selections[0].shape[:2], dtype=np.bool)
    for sub_select in selections:
        if isinstance(sub_select, imgmask):
            sub_select = sub_select.select
        sel = chooser(sel, sub_select)
    return imgmask(img, sel.astype(np.uint8))


mask = partial(_choose, np.logical_and, np.ones)
mask_or = partial(_choose, np.logical_or, np.zeros)


# This namedtuple holds a 3D of the image itself and a 2D array of the
# selected pixels. It gets passed around to effect functions.
imgmask = namedtuple("img", ["img", "select"])


################
# Moving pixels


def flipped(fn):
    def wrapper(masked_img, *args, **kwargs):
        img, select = masked_img
        _rotated = imgmask(np.rot90(img), np.rot90(select))
        for _ in fn(_rotated, *args, **kwargs):
            yield img
    return wrapper


def run_forever(fn, masked_img, *args, **kwargs):
    while True:
        fn(masked_img, *args, **kwargs)
        yield masked_img.img


def mover(transform, fn, *args, **kwargs):
    return transform(fn(*args, **kwargs))


def move_chunks(move_fn, moves):
    for img, select, travel in moves:    
        move_fn(img, select, travel)


rubix_chunks = partial(move_chunks, move_rubix)
swap_chunks = partial(move_chunks, move_swap)


def move_chunks_back(move_fn, moves):
    backward_moves = ((a, b, -travel) for a, b, travel in moves)
    return move_fn(backward_moves)


move_forward_rubix = partial(mover, rubix_chunks)
move_forward_swap = partial(mover, swap_chunks)
move_backward_rubix = partial(
    mover, partial(move_chunks_back, move_forward_rubix))
move_backward_swap = partial(
    mover, partial(move_chunks_back, move_forward_swap))


