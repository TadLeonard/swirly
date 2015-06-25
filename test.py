
import imread
import numpy as np

import swirl


def img():
    return imread.imread("examples/gelface.jpg")


def green_img():
    a = np.ndarray((1000, 1000, 3))
    a[:] = [0, 255, 0]
    return a
 

def test_get_channel_hue():
    i = green_img()
    i[50, :] = [0, 0, 0]  # black horizontal stripe
    i[:, 50] = [0, 0, 0]  # black vertical stripe
    hsl_green = swirl.hsl_filter(h=(110, 130))
    c = swirl.get_channel(i, hsl_green)
    assert not c[50]
    assert np.all(c[:50])
    assert np.all(c[51:]) 


def test_get_channel_lightness():
    i = green_img()
    i[50, :] = [0, 0, 0]  # black horizontal stripe
    i[:, 50] = [0, 0, 0]  # black vertical stripe
    hsl_green = swirl.hsl_filter(l=(85, 89))
    c = swirl.get_channel(i, hsl_green)
    assert not c[50]
    assert np.all(c[:50])
    assert np.all(c[51:]) 


def test_get_channel_all():
    """Test a filter with all three properties set"""
    i = green_img()
    i[50, :] = [0, 0, 0]  # black horizontal stripe
    i[:, 50] = [0, 0, 0]  # black vertical stripe
    hsl_green = swirl.hsl_filter((110, 130), (99, 100), (85, 89))
    c = swirl.get_channel(i, hsl_green)
    assert not c[50]
    assert np.all(c[:50])
    assert np.all(c[51:]) 


def test_get_channel_one_broken():
    """If ONE of the three HSL properties is off, the filter should
    not select the row"""
    i = green_img()
    i[50, :] = [0, 0, 0]  # black horizontal stripe
    i[:, 50] = [0, 0, 0]  # black vertical stripe
    hsl_green = swirl.hsl_filter((110, 130), (48, 52), (85, 86))  # L is wrong
    c = swirl.get_channel(i, hsl_green)
    assert not c[50]
    assert not np.all(c[:50])
    assert not np.all(c[51:]) 

