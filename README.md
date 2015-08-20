# swirly
This is an art collaboration with [Rollin Leonard](www.rollinleonard.com).

## Selecting pixels

The software uses [husl-numpy](https://github.com/TadLeonard/husl-numpy) to select
pixels based on **H**ue, **S**aturation, and/or **L**ightness values based on the
[HUSL color space](www.husl-colors.org). Selected pixels are then moved in various
ways.

## Moving pixels

1. Rubix method: shift a row/column of pixels with a wrap-around effect
2. Swap method: move a selection by swapping their position with interfering pixels

## Installation

From source: `python setup.py build_ext --inplace`
With pip: `pip install git+https:github.com/TadLeonard/swirly`
