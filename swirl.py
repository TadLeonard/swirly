#!/usr/bin/env python
# coding: utf-8

import sys
import random
import logging
import math

from collections import namedtuple
from functools import partial
from itertools import takewhile, repeat, islice, zip_longest

from imread import imread, imwrite
import numpy as np
from moviepy.editor import VideoClip
import nphusl


logging.basicConfig(level=logging.INFO)


###################
# Filtering pixels


_props = namedtuple("channel", "h s l")


def hsl_filter(h=(), s=(), l=()):
    """Returns a namedtuple of properties that are used to find
    'channels' of a certain hue, saturation, and/or lightness in an image.
    If the user wants to filter for light rows/columns of pixels, they might
    use `hsl_filter(l=[90.5, 99.0])."""
    h, s, l = (tuple(p) for p in [h, s, l])  # let's make them all tuples

    # let's make sure the user has specified (min, max) for HSL
    if any(len(p) not in (0, 2) for p in (h, s, l)):
        raise ValueError("HSL filters can only be tuples like (min, max)")

    # let's make sure min, max values fit within their appropriate range
    if h:
        h = _valid_h(*h)
    if s:
        s = _valid_s(*s)
    if l:
        l = _valid_l(*l)

    return _props(h, s, l)


def _valid_h(hmin, hmax):
    return (max(hmin, 0), max(hmax, 0))


def _valid_s(smin, smax):
    return (min(max(smin, 0), 100),
            min(max(smax, 0), 100))


_valid_l = _valid_s  # saturation has the same range as lightness


def get_channel(img, filter_hsl, avg_husl):
    """Returns row indices for which the HSL filter constraints are met"""
    idx_select = np.ones(img.shape[0], dtype=bool)  # no "rows" selected initially
    for prop_idx, prop in enumerate(filter_hsl):
        if not prop:
            continue
        pmin, pmax = prop
        avg = avg_husl[:, prop_idx]
        # add/subtract 0.1 for convenient comparison of numpy arrays of floats
        # i.e. we want 100 to be not < 100.0000000000023
        idx_select[avg < (pmin - 0.1)] = 0
        idx_select[avg > (pmax + 0.1)] = 0
    return idx_select


################
# Moving pixels


@profile
def move(i, travel):
    """Shift `img` a distance of `travel` in the positive direction.
    The pixel array wraps around, so the last pixels will end up being
    the first pixels."""
    # NOTE: This copy is needed. Doing a backwards slice assignment is a good
    # workaround for an in place shift (numpy.roll creates a copy!), but it
    # is a real hack. Backwards slice assignment will work with c array
    # ordering, for example, but will break for Fortran style arrays.
 #   if len(img.shape) > 2:
 #       img = img.swapaxes(0, 1)
    #i = img.squeeze()
    if travel < 0:
        tail = i[:-travel].copy()
        i[:travel] = i[-travel:]
        i[travel:] = tail
    else:
        tail = i[-travel:].copy()  # pixel array wraps around
        i[travel:] = i[:-travel]  # move bulk of pixels
        i[:travel] = tail  # move the saved `tail` into vacated space
    return i


def mass_move(img, travel):
    if travel < 0:
        tail = img[:-travel].copy()


def flipped(fn):
    def wrapper(img, select, *args, **kwargs):
        _img = np.rot90(img)
        _select = np.rot90(select)
        for _ in fn(_img, _select, *args, **kwargs):
            yield img
    return wrapper


def run_while_changed(fn, img, *args, **kwargs):
    changed = True
    while changed:
        changed = fn(img, *args, **kwargs)
        yield img


@profile
def clump_cols(img, select, moves):
    rwhere, cwhere = np.nonzero(select)
    total_avg = np.mean(rwhere)
 #   img[:] = 255
 #   img[total_avg, :] = 255, 0, 255
    cols = np.unique(cwhere)
    if len(moves) == 1:
        travels = np.zeros((cols.size,))
        travels[:] = moves[0]
    else: 
        travels = np.random.choice(moves, cols.size)
    nz = np.nonzero(travels)
    cols, travels = cols[nz], travels[nz]

    s = select[:, cols]
    row_indices = np.arange(0, img.shape[0]) 
    row_matrix = np.zeros(select.shape, dtype=np.float)
    rm = row_matrix.swapaxes(0, 1)
    rm[:, None] = row_indices
    row_matrix = rm.swapaxes(0, 1)
    row_matrix[~select] = np.nan
    col_avgs = np.nanmean(row_matrix, axis=0)

    diff = col_avgs - total_avg
    abs_diff = np.abs(diff)

    # when columns are as close the median as they can be, we want them to
    # gradually move with a random choice between moving 0 or 1 pixels
    stuck = abs_diff < travels
    travels[stuck] = np.random.choice((0, 1), np.count_nonzero(stuck))
    travels[diff > 0] *= -1  # reverse row travel dir if row avg < total avg
    nz = np.nonzero(travels)
    travels = travels[nz]
    cols = cols[nz]

#    for col, travel in zip(cols, travels):
        #heights = rwhere[cwhere == col]
        #col_avg = np.mean(heights)
        #abs_diff = abs(col_avg - total_avg)
        #if abs_diff < travel:
        #    travel = 1
        #if col_avg > total_avg:
        #    travel = -travel
#        move(img[:, col], travel)
#        move(select[:, col], travel)


#    return True

    #img[:] = 255
    #img[select] = 255, 0, 255

    all_travels = tuple(-m for m in moves) + moves
#    for travel in all_travels:
#        cols_to_move = cols[travels == travel]
#        if not cols_to_move.size:
#            continue
#        for buf in _chunk_select_2(cols_to_move):
#            for col in buf:
#                move(img[:, col], travel) 
#                move(select[:, col], travel)
#    return True
                

    moved = 0
    for travel in all_travels:
        cols_to_move = cols[travels == travel]
        moved += cols_to_move.size
        if not cols_to_move.size:
            continue
        for start, stop in _chunk_select(cols_to_move):
#            for col in range(start, stop):
#                move(img[:, col], travel) 
#                move(select[:, col], travel)
            move(img[:, start: stop], travel)
            move(select[:, start: stop], travel)
    #return False
    return True


@profile
def _chunk_select(rows):
    """Generate contiguous chunks of indices in tuples of
    (start_index, stop_index) where stop_index is not inclusive"""
    contiguous = np.diff(rows) == 1 
    buf = []
    for r, n in zip_longest(rows, contiguous):
        buf.append(r)
        if not n:
            yield buf[0], buf[-1] + 1
            buf = []
    if buf:
        yield buf[0], buf[-1] + 1

   
clump_vert = partial(run_while_changed, clump_cols)
clump_horz = flipped(clump_vert)


def clump_cols2(img, select, moves):
    rwhere, cwhere = np.nonzero(select)
    total_avg = np.mean(rwhere)
#    img[:] = 255
#    img[total_avg, :] = 255, 0, 255
    cols = np.unique(cwhere)
    if len(moves) == 1:
        travels = np.zeros((len(cols),))
        travels[:] = moves[0]
    else: 
        travels = np.random.choice(moves, len(cols))
    nz = np.nonzero(travels)
    cols, travels = cols[nz], travels[nz]
    for col, travel in zip(cols, travels):
        heights = rwhere[cwhere == col]
        col_avg = np.mean(heights)
#        img[col_avg, col] = 255, 0, 100
        abs_diff = abs(col_avg - total_avg)
        if abs_diff < travel:
            travel = 1
        if col_avg > total_avg:
            travel = -travel
        move(img[:, col], travel)
        move(select[:, col], travel)
    return True
    return False


clump_vert2 = partial(run_while_changed, clump_cols2)
clump_horz2 = flipped(clump_vert2)


#disperse_vert = partial(run_while_changed, disperse_cols)
#disperse_horz = flipped(disperse_vert)


def fuzz_horz(img, select, moves=(0, 1)):
    moves = tuple(moves)
    fuzz_moves = tuple(-m for m in moves if m) + moves
    while True:
        rows = np.nonzero(np.any(select, axis=1))[0]
        fuzz_rows(img, select, rows, fuzz_moves) 
        yield img


fuzz_vert = flipped(fuzz_horz)


def fuzz_rows(img, select, rows, moves):
    travels = np.random.choice(moves, len(rows))
    for row, travel in zip(rows, travels):
        if not travel:
            continue
        move(img[row, :], travel)
        move(select[row, :], travel)


######################
# Creating animations
    

def read_img(path):
    img = np.squeeze(imread(path))
    if len(img.shape) == 2:
        _img = np.ndarray(img.shape + (3,), dtype=img.dtype)
        _img[:] = img[..., None]
        img = _img
    logging.info("Initial image shape: {}".format(img.shape))
    logging.info("Working image shape: {}".format(img.shape))
    return img


def dark_clump_turns(img, select, moves):
    yield img
    while True:
        for _ in clump_vert(img, select, moves):
            yield img
        for _ in clump_horz(img, select, moves):
            yield img


def zip_effects(img, *effects):
    yield img
    for _ in zip(*effects):
        yield img


def frame_maker(effects):
    def make(_):
        return next(effects)
    return make


#################
# Whole programs


def clump_dark(filename, percentile=4.0):
    img = read_img(filename)
    hsl = nphusl.to_husl(img)
    _, _, L = (hsl[..., n] for n in range(3))
    dark = L < np.percentile(L, 4.0)
    logging.info("Selection ratio: {:1.1f}%".format(
                 100 * np.count_nonzero(dark) / dark.size))
    travel = (1,)
   # img[dark] = 255, 0, 255
    vert = clump_vert(img, dark, travel)
    horz = clump_horz(img, dark, travel)
    return zip_effects(img, vert)


if __name__ == "__main__":
    infile, outfile = sys.argv[1: 3]
    frames = clump_dark(infile, 4.0)
    make_frame = frame_maker(frames)
    animation = VideoClip(make_frame, duration=10)
    animation.write_videofile(outfile, fps=24, audio=False, threads=2)

