#!/usr/bin/env python
# coding: utf-8

import sys
import random
import math
from collections import namedtuple
from itertools import starmap

from imread import imread
import numpy as np
from moviepy.editor import VideoClip
import nphusl



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


VERTICAL = "vertical"
HORIZONTAL = "horizontal"


def move(img, travel):
    """Shift `img` a distance of `travel` in the positive direction.
    The pixel array wraps around, so the last pixels will end up being
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
        tail = img[-travel:].copy()  # pixel array wraps around
        img[travel:] = img[:-travel]  # move bulk of pixels
        img[:travel] = tail  # move the saved `tail` into vacated space
    return img


class FilterEffect:

    def __init__(self, hsl_filter, pattern):
        self.hsl_filter = hsl_filter
        self.pattern = pattern
        self.iteration = 0

    def move(self, img, avg_husl):
        props = next(self.pattern)
        if props.direction == VERTICAL:
            img = img.swapaxes(0, 1)
        self.iteration += 1

    def move_rows(self, img):
        pass



def rand_direction():
    choices = (VERTICAL, HORIZONTAL)
    while True:
        yield random.choice(choices)


def wave_travel(length, mag_coeff=4, period=50, offset=0):
    i = 0
    while True:
        i += 0.1
        travel = mag_coeff * math.cos(math.pi * i / period)
        travel = int(round(travel))
        travel += offset
        prev = travel
        yield travel


def clump_horizontal(img, select, moves=(0, 1)):
    img = np.rot90(img)
    select = np.rot90(select)
    return clump_vertical(img, select, moves)


def clump_vertical(img, select, moves=(0, 1)):
    while True:
        clump_cols(img, select, moves)
        #fuzz_rows(img, select, fuzz_moves, rows)
        yield img


def clump_cols(img, select, moves):
    rwhere, cwhere = np.nonzero(select)
    total_avg = np.mean(rwhere)
    #rows, cols = np.unique(rwhere), np.unique(cwhere)
    cols = np.unique(cwhere)
    if len(moves) == 1:
        travels = np.zeros((len(cols),))
        travels[:] = moves[0]
    else: 
        travels = np.random.choice(moves, len(cols))
    for col, travel in zip(cols, travels):
        if not travel:
            continue
        heights = rwhere[cwhere == col]
        col_avg = np.mean(heights)
        if col_avg > total_avg:
            travel = -travel
        move(img[:, col], travel)
        move(select[:, col], travel)
    #return rows  # rows affected by clumping


def fuzz_horizontal(img, select, moves=(0, 1)):
    moves = tuple(moves)
    fuzz_moves = tuple(-m for m in moves if m) + moves
    while True:
        rows = np.nonzero(np.any(select, axis=1))[0]
        fuzz_rows(img, select, rows, fuzz_moves) 
        yield img


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
    return imread(path)


if __name__ == "__main__":
    img = read_img(sys.argv[1])

    husl = nphusl.to_husl(img)
    H, S, L = (husl[..., n] for n in range(3))
    #lo, hi = 250, 290
    #select = np.logical_and(H > lo, H < hi)
    #select = np.logical_and(select, S > 50)
    select = L < 40
    
    moves = (5,)
    clumps = clump_vertical(img, select, moves)
    clumps2 = clump_horizontal(img, select, moves)
    #fuzzes = fuzz_horizontal(img, select, moves)
    
    def make_frame(_):
        i = next(clumps2)
        i = next(clumps)
        return i
        

    animation = VideoClip(make_frame, duration=7)
    animation.write_videofile("bloop.mp4", fps=24, audio=False, threads=2)
    #animation.write_gif("bloop.gif", fps=24, opt="OptimizePlus")

