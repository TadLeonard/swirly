"""The final product. A combination of effect generators,
image masks, and movers are used to create frame generators."""

import numpy as np
import nphusl

from . import movers
from . import effects


clump = partial(move_forward_rubix, clump_cols)
clump_vert = partial(run_forever, clump)
clump_horz = flipped(clump_vert)
disperse = partial(move_backward_rubix, clump_cols)
disperse_vert = partial(run_forever, disperse)
disperse_horz = flipped(disperse_vert)


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
    light = mask(L > 3)
    blue = mask(img, light, H > 240, H < 290)
    red = mask(img, light, _or(H < 40, H > 320))
    travel = (4,)
    blue_up = slide_vert(blue, travel)
    red_right = slide_horz(red, travel)
    #yield from zip_effects(img, blue_up, red_right)
    light_right = slide_horz(mask(light), travel)
    yield from zip_effects(img, light_right)


##################################
# Creating animations from effects
    

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

