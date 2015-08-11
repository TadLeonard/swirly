#!/usr/bin/env python
# coding: utf-8

import sys
import random
import logging
import math
import warnings

from collections import namedtuple
from functools import partial, wraps
from itertools import takewhile, repeat, islice, zip_longest

from imread import imread, imwrite
import numpy as np
from moviepy.editor import VideoClip
import nphusl
from swirlop import chunk_select, avg_col_height, nonzero2d


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
    idx_select = np.ones(img.shape[0], dtype=np.bool)  # no "rows" selected initially
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


def select(*selections):
    sel = np.ones(selections[0].shape[:2], dtype=np.bool)
    for sub_select in selections:
        sel = np.logical_and(sel, sub_select)
    return sel


################
# Moving pixels


@profile
def move(img, travel):
    """Shift `img` a distance of `travel` in the positive direction.
    The array wraps around, so the last pixels will end up being
    the first pixels."""
    # NOTE: This copy is needed. Doing a backwards slice assignment is a good
    # workaround for an in place shift (numpy.roll creates a copy!), but it
    # is a real hack. Backwards slice assignment will work with c array
    # ordering, for example, but will break for Fortran style arrays.
    if travel < 0:
        tail = img[:-travel].copy()
        img[:travel] = img[-travel:]
        img[travel:] = tail
    else:
        tail = img[-travel:].copy()  # pimgxel array wraps around
        img[travel:] = img[:-travel]  # move bulk of pimgxels
        img[:travel] = tail  # move the saved `taimgl` imgnto vacated space
    return img


def flipped(fn):
    def wrapper(img, select, *args, **kwargs):
        _img = np.rot90(img)
        _select = np.rot90(select)
        for _ in fn(_img, _select, *args, **kwargs):
            yield img
    return wrapper


@profile
def run_forever(fn, img, *args, **kwargs):
    while True:
        fn(img, *args, **kwargs)
        yield img


def mover(transform, fn, *args, **kwargs):
    return transform(fn(*args, **kwargs))


@profile
def move_chunks(moves):
    for arr, travel in moves:
        move(arr, travel)


def move_chunks_back(moves):
    backward_moves = ((arr, -travel) for arr, travel in moves)
    return move_chunks(backward_moves)


move_forward = partial(mover, move_chunks)
move_backward = partial(mover, move_chunks_back)


@profile
def clump_cols(img, select, moves):
    index_data = nonzero2d(select.swapaxes(0, 1).astype(np.uint8))
    col_avgs, cols, total_avg = index_data 
    if not cols:
        return  # no work to do
    col_avgs = np.array(col_avgs, dtype=np.int)
    cols = np.array(cols, dtype=np.int)

    if len(moves) == 1:
        travels = np.zeros((cols.size,))
        travels[:] = moves[0]
    else: 
        travels = np.random.choice(moves, cols.size)

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

    all_travels = tuple(-m for m in moves) + moves
    for travel in all_travels:
        cols_to_move = cols[travels == travel]
        if not cols_to_move.size:
            continue
        for start, stop in chunk_select(cols_to_move):
            yield img[:, start: stop], travel
            yield select[:, start: stop], travel

             
clump = partial(move_forward, clump_cols)
clump_vert = partial(run_forever, clump)
clump_horz = flipped(clump_vert)
disperse = partial(move_backward, clump_cols)
disperse_vert = partial(run_forever, disperse)
disperse_horz = flipped(disperse_vert)


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


def slide_horz(img, select, travel, moves):
    while True:
        rows = np.nonzero(np.any(select, axis=1))[0]
        slide_rows(img, select, rows, moves)


def slide_rows(img, select, rows, moves):
    travels = np.random.choice(moves, rows.size)
    for travel in moves:
        pass


def slide_img_vert(img, travel):
    while True:
        move(img, travel)
        yield img


def slide_img_horz(img, travel):
    flip = np.rot90(img)
    for _ in slide_img_vert(flip, travel):
        yield img
        

######################
# Creating animations
    

def read_img(path):
    img = np.squeeze(imread(path))
    if len(img.shape) == 2:
        _img = np.ndarray(img.shape + (3,), dtype=img.dtype)
        _img[:] = img[..., None]
        img = _img
    elif img.shape[-1] == 4:
        # we can't handle an RGBA array
        img = img[..., :3]
    logging.info("Initial image shape: {}".format(img.shape))
    logging.info("Working image shape: {}".format(img.shape))
    return img


def handle_kb_interrupt(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            yield from fn(*args, **kwargs)
        except KeyboardInterrupt:
            raise StopIteration
    return wrapped


@handle_kb_interrupt
def zip_effects(img, *effects):
    yield img
    for imgs in zip(*effects):
        yield imgs[-1]


@handle_kb_interrupt
def interleave_effects(img, *effects,
                       repeats=1, effects_per_frame=1, rand=False):
    effects = list(effects)
    yield img
    yield from _iterleave(img, effects, repeats, effects_per_frame, rand)


def _iterleave(img, effects, repeats, effects_per_frame, rand):
    count = 0
    prepare = random.shuffle if rand else lambda x: x
    while True:
        prepare(effects)
        for effect in effects:
            for _ in range(repeats):
                e = next(effect)
                count += 1
                if not count % effects_per_frame:
                    yield e


def frame_maker(effects):
    def make(_):
        return next(effects)
    return make


#################
# Whole programs


def clump_dark(img, percentile=4.0):
    hsl = nphusl.to_husl(img)
    H, _, L = (hsl[..., n] for n in range(3))
    dark = L < 5
    logging.info("Selection ratio: {:1.1f}%".format(
                 100 * np.count_nonzero(dark) / dark.size))
    travel = (1,)
    vert = clump_vert(img, dark, travel)
    horz = clump_horz(img, dark, travel)
    yield from zip_effects(img, horz, vert)


def disperse_light(img):
    hsl = nphusl.to_husl(img)
    H, _, L = (hsl[..., n] for n in range(3))
    light = L > 80
    logging.info("Selection ratio: {:1.1f}%".format(
                 100 * np.count_nonzero(light) / light.size))
    travel = (1,)
    vert = disperse_vert(img, light, travel)
    horz = disperse_horz(img, light, travel)
    yield from zip_effects(img, horz, vert)


def clump_hues(img):
    hsl = nphusl.to_husl(img)
    H, _, L = (hsl[..., n] for n in range(3))
    light = L > 1
    travel = (1,)
    
    def effects():
        for selection in select_ranges(H, 50, light):
            yield clump_vert(img, selection, travel)
#        yield slide_img_horz(img, 1)

    yield from zip_effects(img, *effects())


def select_ranges(select_by, percentile, *extra_filters):
    selectable = select(*extra_filters) 
    selectable_values = select_by[selectable]
    min_val = 0
    for max_pct in range(percentile, 100 + percentile, percentile):
        max_val = np.percentile(selectable_values, max_pct)
        selection = select(selectable,
                           select_by < max_val, select_by > min_val)
        yield selection
        min_val = max_val


def blueb(img):
    hsl = nphusl.to_husl(img)
    H, _, L = (hsl[..., n] for n in range(3))
    dark = L < 5
    bright = L > 80
    travel = (1,)
    blue = select(H > 240, H < 290)
    
    hblue = clump_horz(img, bright, travel)
    vblue = clump_vert(img, bright, travel)
    hdark = disperse_horz(img, dark, travel)
    vdark = disperse_vert(img, dark, travel)
    yield from zip_effects(img, hblue, vblue, hdark, vdark)


if __name__ == "__main__":
    infile, outfile = sys.argv[1: 3]
    img, metadata = imread(infile, return_metadata=True)
    frames = clump_dark(img)
    make_frame = frame_maker(frames)
    animation = VideoClip(make_frame, duration=6)
    animation.write_videofile(outfile, fps=24, audio=False, threads=2,
                              preset="ultrafast")
    imwrite("_{}_last.jpg".format("swirl"), img, metadata=metadata,
            opts={"quality": 100})


