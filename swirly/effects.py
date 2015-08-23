"""Various ways to mutate images implemented as enerators that yield the
frames of videos."""

import random
import numpy as np
import nphusl


def slide_cols(masked_img, moves):
    cols = np.nonzero(np.any(masked_img.select, axis=1))[0]
    travels = np.random.choice(moves, cols.size)
    yield from _gen_contiguous_moves(masked_img, travels, cols, moves)


def clump_cols(masked_img, moves):
    col_avgs, total_avg, cols = _get_column_data(masked_img.select)
    if not cols.size:
        return  # no work to do
    travels = _prepare_column_travels(cols, moves)
    clumped = _get_clumped_cols(col_avgs, total_avg, travels)
    travels, cols = _travel_direction(clumped, travels, cols)
    yield from _gen_contiguous_moves(masked_img, travels, cols, moves)


def _get_column_data(select):
    index_data = column_avgs(select.swapaxes(0, 1).astype(np.uint8))
    col_avgs, total_avg = index_data 
    cols = np.nonzero(col_avgs >= 0)[0]
    col_avgs = col_avgs[cols]
    return col_avgs, total_avg, cols


def _prepare_column_travels(cols, moves):
    # create array of random choices from the given moves
    if len(moves) == 1:
        travels = np.zeros((cols.size,))
        travels[:] = moves[0]
    else: 
        travels = np.random.choice(moves, cols.size)
    return travels


def _get_clumped_cols(col_avgs, total_avg, travels):
    diff = col_avgs - total_avg
    abs_diff = np.abs(diff)
    return abs_diff < travels, diff


def _travel_direction(stopped, travels, cols):
    stuck, diff = stopped
    travels[stuck] = np.random.choice((0, 1), np.count_nonzero(stuck))
    travels[diff > 0] *= -1  # reverse row travel dir if row avg < total avg
    nz = np.nonzero(travels)
    return travels[nz], cols[nz]


def _gen_contiguous_moves(masked_img, travels, cols, moves):
    img, select = masked_img
    all_travels = set(-m for m in moves) | set(moves) | {1, -1}
    for travel in all_travels:
        cols_to_move = cols[travels == travel]
        if not cols_to_move.size:
            continue
        for start, stop in chunk_select(cols_to_move):
            yield img[:, start: stop], select[:, start: stop], travel


def fuzz_horz(masked_img, move_magnitudes=(0, 1)):
    fuzz_moves = set(move_magnitudes) | {-m for m in move_magnitudes}
    rows = np.nonzero(np.any(masked_img.select, axis=1))[0]
    travels = np.random.choice(moves, rows.size)
    chunks = _gen_contiguous_moves(masked_img, travels, rows, move_magnitudes)
    yield from chunks

