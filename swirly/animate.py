"""The final product. A combination of effect generators,
image masks, and movers are used to create frame generators."""

from abc import abstractmethod, ABCMeta
from collections import namedtuple
from functools import wraps
import random

from . import _swirlop


def noop_effect(masked_img, moves):
    img, select = masked_img
    yield img, select, moves[0]


def noop_prepare(thing_to_prepare):
    return thing_to_prepare


_view = namedtuple("_view", ["masked_view", "frame"])


def make_view(masked_img, prepare_img=noop_prepare):
    frame = masked_img.img
    prepared = prepare_img(masked_img)
    return _view(prepared, frame)
   

def make_effect(suggest_moves=noop_effect, prepare_moves=noop_prepare):
    def _prepare_effect(masked_view, move_magnitudes):
        suggested_moves = suggest_moves(masked_view, move_magnitudes)
        return prepare_moves(suggested_moves)
    return _prepare_effect


def make_animation(masked_img, move_magnitudes,
                   view=make_view, effect=make_effect,
                   animation=None):
    if animation is None:
        raise NotImplementedError("No animation type specified")
    _view = view(masked_img)
    _effect = effect()
    _animation = animation(_view, _effect, move_magnitudes)
    return _animation

 
class Animation(metaclass=ABCMeta):

    def __init__(self, view, effect, move_magnitudes):
        self.view = view
        self.masked_view = view.masked_view
        self.effect = effect
        self.move_magnitudes = move_magnitudes
        assert all(m > 0 for m in move_magnitudes), "positive magnitudes only"

    def __iter__(self):
        while True:
            yield self.make_frame()

    def make_frame(self, *_):  # discard MoviePy's time argument
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
    move = _swirlop.move_rubix


class SwapAnimation(Animation):
    move = _swirlop.move_swap


### Merging multiple effects into a single animation

def handle_kb_interrupt(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            yield from fn(*args, **kwargs)
        except KeyboardInterrupt:
            return  # no longer raise StopIteration in 3.5+
    return wrapped


class Group:

    def __init__(self, initial_img, *animations):
        self.initial_img = initial_img
        self.animations = animations

    def zip(self):
        iterator = self._zip()
        def make_frame(_):
            return next(iterator)
        return make_frame

    def interleave(self, repeats=1, effects_per_frame=1, rand=False):
        iterator = self.interleave_effects(repeats, effects_per_frame, rand)
        def make_frame(_):
            return next(iterator)
        return make_frame

    @handle_kb_interrupt
    def _zip(self):
        yield self.initial_img
        while True:
            for imgs in zip(*self.animations):
                yield imgs[-1]

    @handle_kb_interrupt
    def _interleave(self, repeats, effects_per_frame, rand):
        yield self.initial_img
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

