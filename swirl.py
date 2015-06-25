#!/usr/bin/env python
# coding: utf-8


import sys
import random
from collections import namedtuple
from itertools import starmap

from imread import imread
import numpy as np
from moviepy.editor import VideoClip
from husl import rgb_to_husl


_props = namedtuple("channel", "h s l")


def hsl_filter(h=(), s=(), l=()):
    """Returns a namedtuple of properties that are used to find
    'channels' of a certain hue, saturation, and/or lightness in an image.
    If the user wants to filter for light rows/columns of pixels, they might
    use `hsl_filter(l=[90.5, 99.0])."""
    h, s, l = [tuple(sorted(p)) for p in [h, s, l]]
    if any(len(p) not in (0, 2) for p in (h, s, l)):
        raise ValueError("HSL filters can only be tuples like (min, max)")
    return _props(h, s, l)


#@profile
def get_channel(img, hsl):
    """Returns indices for which the HSL filter constraints are met"""
    idx_select = np.zeros(img.shape[0])  # no "rows" selected initially
    avg_rgb = np.average(img, axis=1)
    avg_rgb /= 255.0
    avg_husl = np.array(list(starmap(rgb_to_husl, avg_rgb)))


def move(img_pixels, travel):
    """Shift `img_pixels` a distance of `travel` in the positive direction.
    The pixel array wraps around, so the last pixels will end up being
    the first pixels."""
    # NOTE: This copy is needed. Doing a backwards slice assignment is a good
    # workaround for an in place shift (numpy.roll creates a copy!), but it
    # is a real hack. Backwards slice assignment will work with c array
    # ordering, for example, but will break for Fortran style arrays.
    tail = img_pixels[-travel:].copy()  # pixel array wraps around
    img_pixels[travel:] = img_pixels[:-travel]  # move bulk of pixels
    img_pixels[:travel] = tail  # move the saved `tail` into vacated space


def _move_random(img):
    for row in img:
        move(row, random.choice(list(range(1, 5))))
    return img


def animate(img, make_frame, duration):
    animation = VideoClip(make_frame, duration=duration)
    animation.write_gif("bloop.gif", fps=24, opt="OptimizePlus")


def _read_img(path):
    return imread(path)


if __name__ == "__main__":
    img = _read_img(sys.argv[1])
    get_channel(img, hsl_filter)

    def rand_move(t):
        return _move_random(img)
    
    animate(img, rand_move, duration=3)

