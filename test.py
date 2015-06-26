import imread
import numpy as np

import swirl


def img():
    return imread.imread("examples/gelface.jpg")


def green_img():
    a = np.ndarray((1000, 1000, 3))
    a[:] = [0, 255, 0]
    return a
 

def green_channel(*filter_args, **filter_kwargs):
    i = green_img()
    i[50, :] = [0, 0, 0]  # black horizontal stripe
    i[:, 50] = [0, 0, 0]  # black vertical stripe
    filter_hsl = swirl.hsl_filter(*filter_args, **filter_kwargs)
    avg_husl = swirl.get_avg_husl(i)
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

