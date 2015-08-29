from collections import namedtuple
from functools import partial

from . import animate
from . import effects
from .animate import Group
from .animate import make_view, make_effect, make_animation

import numpy as np
import nphusl


### Filtering pixels

def _choose(chooser, starter, img, *selections):
    if not selections:
        return imgmask(img, np.ones(img.shape, dtype=np.uint8))
    selections = tuple(s.select if isinstance(s, imgmask) else s
                       for s in selections)
    sel = starter(selections[0].shape[:2], dtype=np.bool)
    for sub_select in selections:
        sel = chooser(sel, sub_select)
    return imgmask(img, sel.astype(np.uint8))


imgmask = namedtuple("imgmask", ["img", "select"])
_or, _and = np.logical_or, np.logical_and
mask = partial(_choose, _and, np.ones)
mask_or = partial(_choose, _or, np.zeros)


### Preparer functions for views, effects, and animations

def reverse(chunks):
    return ((i, s, -travel) for i, s, travel in chunks)


def flip(masked_img):
    img, select = masked_img
    rotated = imgmask(np.rot90(img), np.rot90(select))
    return rotated


### Specific views, effects, & animations for end user

flipped_view = partial(make_view, prepare_img=flip)
clump_effect = partial(make_effect, effects.clump_cols)
disperse_effect = partial(clump_effect, prepare_moves=reverse)

slide_vert = partial(make_animation, animation=animate.SwapAnimation)
slide_horz = partial(slide_vert, view=flipped_view)
rubix_vert = partial(make_animation, animation=animate.RubixAnimation)
rubix_horz = partial(rubix_vert, view=flipped_view)

clump_vert = partial(rubix_vert, effect=clump_effect)
clump_horz = partial(rubix_horz, effect=clump_effect)

disperse_vert = partial(rubix_vert, effect=disperse_effect)
disperse_horz = partial(disperse_vert, view=flipped_view)


### Complete animations
### These are the "end products"

def clump_dark(img):
    hsl = nphusl.to_husl(img)
    H, L = hsl[..., 0], hsl[..., 2]
    dark = mask(img, L < 5)
    travel = (5,)
    vert = clump_vert(dark, travel)
    horz = clump_horz(dark, travel)
    return Group(img, vert, horz).zip(effects_per_frame=5)


def clump_gradient(img):
    hsl = nphusl.to_husl(img)
    H, L = hsl[..., 0], hsl[..., 2]
    dark = mask(img, L < 5)
    travel = (5,)
    ranges = list(range(1, 5))
    
    def effects(ranges):
        min_L = ranges[0]
        for max_L in ranges[1:]:
            m = mask(img, L < max_L, L > min_L)
            yield clump_vert(m, travel)
            yield clump_horz(m, travel)
            min_L = max_L

    effs = effects(ranges)
    return Group(img, *effs).zip(effects_per_frame=5)


def disperse_light(img):
    hsl = nphusl.to_husl(img)
    H, L = hsl[..., 0], hsl[..., 2]
    light = mask(img, L > 80)
    travel = (1,)
    vert = disperse_vert(img, light, travel)
    horz = disperse_horz(img, light, travel)
    return Group(img, horz, vert).zip()


def clump_hues(img):
    hsl = nphusl.to_husl(img)
    H, L = hsl[..., 0], hsl[..., 2]
    light = mask(img, L > 1)
    travel = (1,)
    effects = (clump_vert(s, travel) for s in _select_ranges(H, 50, light))
    return Group(img, *effects()).zip()


def _select_ranges(select_by, percentile, *extra_filters):
    selectable = mask(select_by, *extra_filters) 
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
    light = mask(img, L > 4)
    blue = mask(img, light, H > 240, H < 290)
    red = mask(img, light, _or(H < 40, H > 320))
    travel = (4,)
    blue_up = slide_vert(blue, travel)
    red_right = slide_horz(red, travel)
    light_right = slide_horz(light, travel)
    return Group(img, light_right).zip()


