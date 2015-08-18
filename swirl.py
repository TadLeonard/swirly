#!/usr/bin/env python
# coding: utf-8

import argparse
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
from swirlop import chunk_select, column_avgs, move_rubix


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


def move_chunks(moves):
    for img, select, travel in moves:    
        move_rubix(img, select, travel)


def move_chunks_back(moves):
    backward_moves = ((a, b, -travel) for a, b, travel in moves)
    return move_chunks(backward_moves)


move_forward = partial(mover, move_chunks)
move_backward = partial(mover, move_chunks_back)


imgmask = namedtuple("img", ["img", "select"])


def clump_cols(masked_img, moves):
    col_avgs, total_avg, cols = _get_column_data(masked_img.select)
    if not cols.size:
        return  # no work to do
    travels = _prepare_column_travels(cols, moves)
    clumped = _get_clumped_cols(col_avgs, total_avg, travels)
    travels, cols = _travel_direction(clumped, travels, cols)
    yield from _gen_contiguous_moves(masked_img, travels, cols, moves)


def _get_column_data(select):
    index_data = column_avgs(select.swapaxes(0, 1).astype(np.uint8))
    col_avgs, total_avg = index_data 
    cols = np.nonzero(col_avgs >= 0)[0]
    col_avgs = col_avgs[cols]
    return col_avgs, total_avg, cols


def _prepare_column_travels(cols, moves):
    # create array of random choices from the given moves
    if len(moves) == 1:
        travels = np.zeros((cols.size,))
        travels[:] = moves[0]
    else: 
        travels = np.random.choice(moves, cols.size)
    return travels


def _get_clumped_cols(col_avgs, total_avg, travels):
    diff = col_avgs - total_avg
    abs_diff = np.abs(diff)
    return abs_diff < travels, diff


def _travel_direction(stopped, travels, cols):
    stuck, diff = stopped
    travels[stuck] = np.random.choice((0, 1), np.count_nonzero(stuck))
    travels[diff > 0] *= -1  # reverse row travel dir if row avg < total avg
    nz = np.nonzero(travels)
    return travels[nz], cols[nz]


def _gen_contiguous_moves(masked_img, travels, cols, moves):
    img, select = masked_img
    all_travels = set(-m for m in moves) | set(moves) | set((1, -1))
    for travel in all_travels:
        cols_to_move = cols[travels == travel]
        if not cols_to_move.size:
            continue
        for start, stop in chunk_select(cols_to_move):
            yield img[:, start: stop], select[:, start: stop], travel


clump = partial(move_forward, clump_cols)
clump_vert = partial(run_forever, clump)
clump_horz = flipped(clump_vert)
disperse = partial(move_backward, clump_cols)
disperse_vert = partial(run_forever, disperse)
disperse_horz = flipped(disperse_vert)


def fuzz_horz(masked_img, moves=(0, 1)):
    moves = tuple(moves)
    fuzz_moves = tuple(-m for m in moves if m) + moves
    while True:
        rows = np.nonzero(np.any(masked_img.select, axis=1))[0]
        fuzz_rows(masked_img, rows, fuzz_moves) 
        yield masked_img.img


fuzz_vert = flipped(fuzz_horz)


def fuzz_rows(masked_img, rows, moves):
    travels = np.random.choice(moves, len(rows))
    img, select = masked_img
    for row, travel in zip(rows, travels):
        if not travel:
            continue
        move_rubix(img[row, :], travel)
        move_rubix(select[row, :], travel)


def slide_cols(masked_img, moves):
    cols = np.nonzero(np.any(masked_img.select, axis=1))[0]
    travels = np.random.choice(moves, cols.size)
    yield from _gen_contiguous_moves(masked_img, travels, cols, moves)


slide = partial(move_backward, slide_cols)
slide_vert = partial(run_forever, slide)
slide_horz = flipped(slide_vert)


def slide_img_vert(img, travel):
    while True:
        move_rubix(img, travel)
        yield img


def slide_img_horz(img, travel):
    flip = np.rot90(img)
    for _ in slide_img_vert(flip, travel):
        yield img
        

######################
# Creating animations
    

def read_img(path, return_metadata=False):
    img = np.squeeze(imread(path, return_metadata=return_metadata))
    if return_metadata:
        img, metadata = img
    if len(img.shape) == 2:
        _img = np.ndarray(img.shape + (3,), dtype=img.dtype)
        _img[:] = img[..., None]
        img = _img
    elif img.shape[-1] == 4:
        # we can't handle an RGBA array
        img = img[..., :3]
    logging.info("Initial image shape: {}".format(img.shape))
    logging.info("Working image shape: {}".format(img.shape))
    if return_metadata:
        return img, metadata
    else:
        return img


def handle_kb_interrupt(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            yield from fn(*args, **kwargs)
        except KeyboardInterrupt:
            return  # no longer raise StopIteration in 3.5+
            #raise StopIteration
    return wrapped


@handle_kb_interrupt
def zip_effects(first_img, *effects):
    yield first_img
    for imgs in zip(*effects):
        yield imgs[-1]


@handle_kb_interrupt
def interleave_effects(first_img, *effects,
                       repeats=1, effects_per_frame=1, rand=False):
    yield first_img
    effects = list(effects)
    yield from _iterleave(effects, repeats, effects_per_frame, rand)


def _iterleave(effects, repeats, effects_per_frame, rand):
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


def clump_dark(img):
    hsl = nphusl.to_husl(img)
    H, _, L = (hsl[..., n] for n in range(3))
    dark = mask(img, L < 5)
    travel = (5,)
    vert = clump_vert(dark, travel)
    horz = clump_horz(dark, travel)
    yield from zip_effects(img, horz, vert)


def disperse_light(img):
    hsl = nphusl.to_husl(img)
    H, _, L = (hsl[..., n] for n in range(3))
    light = mask(img, L > 80)
    logging.info("Selection ratio: {:1.1f}%".format(
                 100 * np.count_nonzero(light) / light.size))
    travel = (1,)
    vert = disperse_vert(img, light, travel)
    horz = disperse_horz(img, light, travel)
    yield from zip_effects(img, horz, vert)


def clump_hues(img):
    hsl = nphusl.to_husl(img)
    H, _, L = (hsl[..., n] for n in range(3))
    light = mask(img, L > 1)
    travel = (1,)
    
    def effects():
        for selection in select_ranges(H, 50, light):
            yield clump_vert(selection, travel)

    yield from zip_effects(img, *effects())


def select_ranges(select_by, percentile, *extra_filters):
    selectable = mask(img, *extra_filters) 
    selectable_values = select_by[selectable.select]
    min_val = 0
    for max_pct in range(percentile, 100 + percentile, percentile):
        max_val = np.percentile(selectable_values, max_pct)
        selection = mask(selectable.select,
                         select_by < max_val, select_by > min_val)
        yield selection
        min_val = max_val


def slide_colors(img):
    hsl = nphusl.to_husl(img)
    H, L = hsl[..., 0], hsl[..., 2]
    light = L > 3
    blue = mask(img, light, H > 240, H < 290)
    red = mask(img, light, _or(H < 40, H > 320))
    print(np.sum(blue.select), np.sum(red.select))
    travel = (1,)
    blue_up = slide_vert(blue, travel)
    red_right = slide_horz(red, travel)
    yield from zip_effects(img, blue_up, red_right)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="input image")
    parser.add_argument("video", type=str, help="output video")
    args = parser.parse_args()
    img, metadata = read_img(args.image, return_metadata=True)
    #frames = clump_dark(img)
    frames = slide_colors(img)
    make_frame = frame_maker(frames)
    animation = VideoClip(make_frame, duration=60)
    animation.write_videofile(args.video, fps=24, audio=False, threads=1,
                              preset="ultrafast")
    imwrite("_{}_last.jpg".format("swirl"), img, metadata=metadata,
            opts={"quality": 100})


