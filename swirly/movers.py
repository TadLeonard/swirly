"""Functions for moving pixels. These functions carry out the
mutations dictated by effect generators."""

from functools import partial

import numpy as np
from _swirlop import chunk_select, column_avgs, move_rubix, move_swap


################
# Moving pixels

no_op = lambda x: x


def mover(fn, masked_img, *args, **kwargs):
    def move(*args, **kwargs):
        return fn(masked_img, *args, **kwargs)
    return move


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


