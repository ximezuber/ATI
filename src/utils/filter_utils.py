import numpy as np
from src.utils.image_utils import normalize
from src.utils.mask_utils import weighted_mean
from numpy import copy
from math import pi
from scipy import ndimage


def isotropic_dif(img, t):
    new_img = copy(img)
    mask = gaussian_conv(t)

    if len(img.shape) > 2:
        for dim in range(img.shape[-1]):
                convolution = ndimage.convolve(
                    new_img[:, :, dim], mask, mode="constant", cval=0.0
                )
                new_img[:, :, dim] = convolution
    else:
        new_img = ndimage.convolve(
                    new_img, mask, mode="constant", cval=0.0
                )
                

    return normalize(new_img, 0,  weighted_mean(255*np.ones(mask.shape), mask))



def gaussian_conv(t):
    mask_side = 7
    mask = np.ones((mask_side, mask_side))

    
    for i in range(-3, 3):
        for j in range(-3, 3):
            mask[3 + i][3 + j] = (((4 * pi * t) ** (-1)) * np.exp(-(i**2 + j**2) / (4 * t)))

    return mask
    