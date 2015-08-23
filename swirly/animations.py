from functools import partial

from . import animate
from .animate import interleave_effects, zip_effects
from .animate import view, effect, make_animation
from .animate import mask, mask_or

import numpy as np


_or, _and = np.logical_or, np.logical_and


# Specific views, effects, & animations for end user

flipped_view = partial(view, prepare_img=flip)
clump_effect = partial(effect, clump_cols)
disperse_effect = partial(clump_effect, prepare_moves=reverse)


# Complete animation makers

slide_vert = partial(make_animation, animation=animate.SwapAnimation)
slide_horz = partial(slide_vert, view=flipped_view)
rubix_vert = partial(make_animation, animation=animate.RubixAnimation)
rubix_horz = partial(rubix_vert, view=flipped_view)

clump_vert = partial(rubix_vert, effect=clump_effect)
clump_horz = partial(rubix_horz, effect=clump_effect)

disperse_vert = partial(rubix_vert, effect=disperse_effect)
disperse_horz = partial(disperse_vert, view=flipped_view)


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
    light_right = slide_horz(light, travel)
    yield from zip_effects(img, light_right)

