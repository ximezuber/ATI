from numpy import copy

import numpy as np


def mean_filter(img, mask_side):
    new_img = copy(img)
    aux_img = copy(img)
    first_column = img[:, 0]
    last_column = img[:, -1]
    aux_img = np.hstack((first_column[:, np.newaxis], aux_img))
    aux_img = np.hstack((aux_img, last_column[:, np.newaxis]))
    first_row = img[0]
    first_row = np.insert(first_row, 0, np.zeros(img[0][0].shape, dtype=np.uint8), axis=0)
    first_row = np.insert(first_row, len(first_row), np.zeros(img[0][0].shape, dtype=np.uint8), axis=0)
    last_row = img[-1]
    last_row = np.insert(last_row, 0, np.zeros(img[0][0].shape, dtype=np.uint8), axis=0)
    last_row = np.insert(last_row, len(last_row), np.zeros(img[0][0].shape, dtype=np.uint8), axis=0)
    aux_img = np.insert(aux_img, 0, first_row, axis=0)
    aux_img = np.insert(aux_img, len(aux_img), last_row, axis=0)
    for i in range(0, len(img)):
        end_aux_col = i + mask_side
        for j in range(0, len(img[0])):
            end_aux_row = j + mask_side
            if len(img.shape) > 2:
                for k in range(0, len(img[0][0])):
                    new_img[i][j][k] = np.mean(aux_img[i: end_aux_col,
                                               j: end_aux_row, k])
            else:
                new_img[i][j] = np.mean(aux_img[i: end_aux_col,
                                          j: end_aux_row])
    return new_img
