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


#@profile
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
    if travel == 0:
        return
    elif travel < 0:
        if len(img.shape) == 1:
            img = img[::-1]
        else:
            img = img[::, ::-1]
    tail = img[-travel:].copy()  # pixel array wraps around
    img[travel:] = img[:-travel]  # move bulk of pixels
    img[:travel] = tail  # move the saved `tail` into vacated space


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


def clump_vertical(img, select, magnitude=1):
    while True:
        rwhere, cwhere = np.where(select)
        total_avg = np.average(rwhere)
        debug = np.zeros(img.shape, dtype=img.dtype)
        debug[:] = 255
        for col in set(cwhere):
        #    continue
            col_avg = np.average(rwhere[cwhere == col])
            debug[:, col][select[:, col]] = 0, 20, 230
            debug[col_avg-1: col_avg+1, col] = 100, 0, 0
            travel = magnitude if col_avg < total_avg else -magnitude
            if travel < 0:
                print(travel)
            move(img[:, col], travel)
            move(select[:, col], travel)
      #  for row in set(rwhere):
      #      travel = magnitude * random.choice([1, -1])
      #      move(img[row, :], travel)
      #      move(select[row, :], travel)
        debug[total_avg-1: total_avg+1, :] = 255, 0, 0
        yield debug
        #yield img


def rand_travel(lo, hi):
    choices = list(range(lo, hi))
    while True:
        yield random.choices(choices)


######################
# Creating animations
    

def animate_gif(animation):
    animation.write_gif("bloop.gif", fps=24, opt="OptimizePlus")


def animate_mp4(animation):
    animation.write_videofile("bloop.mp4", fps=10, audio=False, threads=2)


def read_img(path):
    return imread(path)


if __name__ == "__main__":
    img = read_img(sys.argv[1])
    hues = nphusl.to_hue(img)
    lo, hi = 250, 290
    select = np.logical_and(hues > lo, hues < hi)
    clumps = clump_vertical(img, select, 1)
    
    def make_frame(_):
        return next(clumps)

    animation = VideoClip(make_frame, duration=20)
    animate_mp4(animation)

