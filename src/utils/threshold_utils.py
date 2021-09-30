from numpy.lib.function_base import copy
from src.utils.image_utils import thresholding

import numpy as np


def global_threshold(img, t):
    delta_t = 255
    new_img = copy(img)

    while(delta_t > 2):
        g1 = np.count_nonzero(new_img < t)
        g2 = np.count_nonzero(new_img >= t)
        sum1 = np.sum(np.where(new_img < t, new_img, 0))
        sum2 = np.sum(np.where(new_img >= t, new_img, 0))
        m1 = sum1 / g1
        m2 = sum2 / g2
        new_t = 0.5 * (m1 + m2)
        delta_t = abs(int(new_t) - int(t))
        t = new_t
    
    return int(t)
