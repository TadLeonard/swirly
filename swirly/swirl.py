#!/usr/bin/env python
# coding: utf-8

import random
import logging
import math
import warnings

from collections import namedtuple
from functools import partial, wraps
from itertools import takewhile, repeat, islice, zip_longest

import numpy as np
import nphusl


clump = partial(move_forward_rubix, clump_cols)
clump_vert = partial(run_forever, clump)
clump_horz = flipped(clump_vert)
disperse = partial(move_backward_rubix, clump_cols)
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


slide = partial(move_backward_swap, slide_cols)
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
        

