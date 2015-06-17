from __future__ import print_function

import sys
import random

import imread
import numpy as np
from moviepy.editor import VideoClip
from husl import rgb_to_husl


def make_grid(img):
    rows = np.sum(img, axis=1)
    cols = np.sum(img, axis=0)


def move(img_pixels, travel):
    """Shift `img_pixels` a distance of `travel` in the positive direction.
    The pixel array wraps around, so the last pixels will end up being
    the first pixels."""
    tail = img_pixels[-travel:].copy()  # pixel array wraps around
    img_pixels[travel:] = img_pixels[:-travel]  # move bulk of pixels
    img_pixels[:travel] = tail  # move the saved `tail` into vacated space


def _move_random(img):
    for row in img:
        move(row, random.choice(range(1, 5)))
    return img


def animate(img, make_frame, duration):
    animation = VideoClip(make_frame, duration=duration)
    animation.write_gif("bloop.gif", fps=24)


if __name__ == "__main__":
    img = imread.imread(sys.argv[1])
    make_grid(img)

    def rand_move(t):
        return _move_random(img)

    animate(img, rand_move, duration=3)
