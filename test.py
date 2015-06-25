
import imread
import numpy as np


def img():
    return imread.imread("examples/gelface.jpg")


def red_img():
    a = np.ndarray((100, 100, 100))
    a[::] = [255, 0, 0]
    return a
 

def test_get_channel():
    i = red_img()
    i[50, :] = [0, 0, 0]  # black horizontal stripe
    

    
    
