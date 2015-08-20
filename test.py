import imread
import numpy as np
import nphusl
import swirl
import swirlop


def img():
    return imread.imread("examples/gelface.jpg")


def green_img():
    a = np.ndarray((100, 100, 3))
    a[:] = [0, 255, 0]
    return a
 

def green_channel(*filter_args, **filter_kwargs):
    i = green_img()
    i[50, :] = [0, 0, 0]  # black horizontal stripe
    i[:, 50] = [0, 0, 0]  # black vertical stripe
    filter_hsl = swirl.hsl_filter(*filter_args, **filter_kwargs)
    husl = nphusl.to_husl(i)
    avg_husl = np.average(husl, axis=0)
    return swirl.get_channel(i, filter_hsl, avg_husl)


def test_get_channel_hue():
    c = green_channel(h=(110, 130))
    assert not c[50]
    assert np.all(c[:50])
    assert np.all(c[51:]) 


def test_get_channel_lightness():
    c = green_channel(l=(85, 89))
    assert not c[50]
    assert np.all(c[:50])
    assert np.all(c[51:]) 


def test_get_channel_all():
    """Test a filter with all three properties set"""
    c = green_channel((110, 130), (99, 100), (85, 89))
    assert not c[50]
    assert np.all(c[:50])
    assert np.all(c[51:]) 


def test_get_channel_one_broken():
    """If ONE of the three HSL properties is off, the filter should
    not select the row"""
    c = green_channel((110, 130), (48, 52), (85, 86))  # L is wrong
    assert not c[50]
    assert not np.all(c[:50])
    assert not np.all(c[51:]) 


def test_row_move():
    a = np.ndarray((100, 3), dtype=np.uint8)
    a[:] = 255
    a[:5] = 0
    swirlop.move_rubix2d(a, 2)
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
    swirlop.move_rubix(s, a, -2)
    b = np.ndarray((100, 3), dtype=np.uint8)
    b[:] = 255
    b[-2:] = 0
    b[:3] = 0
    assert np.all(a == b)


def test_bool_row_move():
    a = np.ndarray((100, 3), dtype=np.uint8)
    a[:] = True
    a[:5] = False
    swirlop.move_rubix2d(a, 2)
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
    
    swirlop.move_rubix(s, z[10, :], -2)
    b = np.ndarray((100, 3), dtype=np.uint8)
    b[:] = True
    b[-2:] = False
    b[:3] = False
    assert np.all(z[10] == b)

