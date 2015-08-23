"""The final product. A combination of effect generators,
image masks, and movers are used to create frame generators."""

from abc import abstractmethod, ABCMeta
from functools import partial

import numpy as np
import nphusl

from . import movers
from . import effects


no_op = lambda x: x
_view = namedtuple("_view", ["masked_view", "frame"])


def view(masked_img, prepare_img=no_op):
    frame = masked_img.img
    prepared = prepare_img(masked_img)
    return _view(prepared, frame)
    

def effect(suggest_moves=no_op, prepare_moves=no_op):
    def _prepare_effect(masked_view, move_magnitudes):
        suggested_moves = suggest_moves(masked_view, move_magnitudes)
        return prepare_moves(suggested_moves)
    return _prepare_effect


class Animation(metaclass=ABCMeta):

    def __init__(self, view, effect, move_magnitudes):
        self.view = view
        self.masked_view = view.masked_view
        self.effect = effect
        self.move_magnitudes = move_magnitudes
        assert all(m > 0 for m in move_magnitudes), "positive magnitudes only"

    def make_frame(self, time):
        self.move_chunks()
        return self.view.frame

    def move_chunks(self):
        effected_chunks = self.effect(self.masked_view, self.move_magnitudes)
        for img, select, travel in effected_chunks:
            self.move(img, select, travel) 

    @abstractmethod
    def move(self, img, select, travel): ...
    

### Concrete animations

class RubixAnimation(Animation):
    move = move_rubix

class SwapAnimation(Animation):
    move = move_swap


### Preparer functions for views, effects, and animations

def reverse(chunks):
    return ((i, s, -travel) for i, s, travel in chunks)


def flip(masked_img):
    img, select = masked_img
    rotated = imgmask(np.rot90(img), np.rot90(select))
    return rotated


def make_animation(masked_img, move_magnitudes,
                   view=view, effect=effect,
                   animation=RubixAnimation):
    _view = view(masked_img)
    _effect = effect()
    _animation = animation(_view, _effect, move_magnitudes)
    return _animation


### Filtering pixels

# This namedtuple holds a 3D of the image itself and a 2D array of the
# selected pixels. It gets passed around to effect functions.
imgmask = namedtuple("img", ["img", "select"])


def _choose(chooser, starter, img, *selections):
    sel = starter(selections[0].shape[:2], dtype=np.bool)
    for sub_select in selections:
        if isinstance(sub_select, imgmask):
            sub_select = sub_select.select
        sel = chooser(sel, sub_select)
    return imgmask(img, sel.astype(np.uint8))


mask = partial(_choose, np.logical_and, np.ones)
mask_or = partial(_choose, np.logical_or, np.zeros)


### Merging multiple effects into a single animation


def handle_kb_interrupt(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            yield from fn(*args, **kwargs)
        except KeyboardInterrupt:
            return  # no longer raise StopIteration in 3.5+
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

