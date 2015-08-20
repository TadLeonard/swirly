import numpy as np
import nphusl
import _swirlop



def test_row_move():
    a = np.ndarray((100, 3), dtype=np.uint8)
    a[:] = 255
    a[:5] = 0
    _swirlop.move_rubix2d(a, 2)
    b = np.ndarray((100, 3), dtype=np.uint8)
    b[:] = 255
    b[2: 7] = 0
    assert np.all(a == b)


def test_neg_row_move():    
    a = np.ndarray((100, 3), dtype=np.uint8)
    s = np.ndarray((100, 3, 3), dtype=np.uint8)
    a[:] = 255
    a[:5] = 0
    s[:] = 0
    s[:5] = 255
    _swirlop.move_rubix(s, a, -2)
    b = np.ndarray((100, 3), dtype=np.uint8)
    b[:] = 255
    b[-2:] = 0
    b[:3] = 0
    assert np.all(a == b)


def test_bool_row_move():
    a = np.ndarray((100, 3), dtype=np.uint8)
    a[:] = True
    a[:5] = False
    _swirlop.move_rubix2d(a, 2)
    b = np.ndarray((100, 3), dtype=np.uint8)
    b[:] = True
    b[2: 7] = False
    assert np.all(a == b)


def test_neg_bool_row_move():    
    s = np.ndarray((100, 100, 3), dtype=np.uint8)
    s[:] = 1
    z = np.ndarray((100, 100, 3), dtype=np.uint8)
    z[:] = True
    z[::, :5] = False
    
    _swirlop.move_rubix(s, z[10, :], -2)
    b = np.ndarray((100, 3), dtype=np.uint8)
    b[:] = True
    b[-2:] = False
    b[:3] = False
    assert np.all(z[10] == b)

