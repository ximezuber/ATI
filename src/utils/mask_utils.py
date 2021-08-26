from numpy import copy

import numpy as np


def fill_outer_matrix(image, mask_side):
    img = copy(image)
    aux_img = copy(img)
    for i in range(0, mask_side // 2):
        first_column = aux_img[:, i]
        last_column = aux_img[:, -i-1]
        img = np.hstack((first_column[:, np.newaxis], img))
        img = np.hstack((img, last_column[:, np.newaxis]))
    aux_img = copy(img)
    for i in range(0, mask_side // 2):
        first_row = aux_img[i]
        last_row = aux_img[-i - 1]
        img = np.insert(img, 0, first_row, axis=0)
        img = np.insert(img, len(img), last_row, axis=0)
    return img


def apply_function_in_mask(img, mask_side, function):
    new_img = copy(img)
    aux_img = fill_outer_matrix(img, mask_side)
    for i in range(0, len(img)):
        end_aux_col = i + mask_side
        for j in range(0, len(img[0])):
            end_aux_row = j + mask_side
            if len(img.shape) > 2:
                for k in range(0, len(img[0][0])):
                    new_img[i][j][k] = function(aux_img[i: end_aux_col, j: end_aux_row, k])
            else:
                new_img[i][j] = function(aux_img[i: end_aux_col, j: end_aux_row])
    return new_img


def mean_filter(img, mask_side):
    return apply_function_in_mask(img, mask_side, np.mean)


def median_filter(img, mask_side):
    return apply_function_in_mask(img, mask_side, np.median)


def weighted_median_filter(img, mask=None):
    if mask is None:
        mask = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    mask_side = len(mask)
    new_img = copy(img)
    aux_img = fill_outer_matrix(img, mask_side)
    for i in range(0, len(img)):
        end_aux_col = i + mask_side
        for j in range(0, len(img[0])):
            end_aux_row = j + mask_side
            if len(img.shape) > 2:
                for k in range(0, len(img[0][0])):
                    new_img[i][j][k] = weighted_median(aux_img[i: end_aux_col, j: end_aux_row, k], mask)
            else:
                new_img[i][j] = weighted_median(aux_img[i: end_aux_col, j: end_aux_row], mask)
    return new_img


def weighted_median(frame, mask):
    median_list = []
    for i in range(0, len(frame)):
        for j in range(0, len(frame[0])):
            for k in range(0, mask[i][j]):
                median_list.append(frame[i][j])
    return np.median(median_list)


def gaussian_filter(img, mask_side, deviation):
    mask = gaussian_mask(mask_side, deviation)
    print(mask)
    new_img = copy(img)
    aux_img = fill_outer_matrix(img, mask_side)
    for i in range(0, len(img)):
        end_aux_col = i + mask_side
        for j in range(0, len(img[0])):
            end_aux_row = j + mask_side
            if len(img.shape) > 2:
                for k in range(0, len(img[0][0])):
                    new_img[i][j][k] = gaussian_mean(aux_img[i: end_aux_col, j: end_aux_row, k], mask)
            else:
                new_img[i][j] = gaussian_mean(aux_img[i: end_aux_col, j: end_aux_row], mask)
    return new_img


def gaussian_mean(frame, mask):
    mean_list = []
    for i in range(0, len(frame)):
        for j in range(0, len(frame[0])):
            for k in range(0, mask[i][j]):
                mean_list.append(frame[i][j])
    return np.mean(mean_list)


def gaussian_mask(mask_side, deviation, mean=0):
    x, y = np.meshgrid(np.linspace(-1, 1, mask_side), np.linspace(-1, 1, mask_side))
    dst = np.sqrt(x * x + y * y)

    gauss_standard = np.exp(-((dst - mean) ** 2 / (2.0 * deviation ** 2)))
    gauss = gauss_standard * 10**decimal_places(gauss_standard[0][0])  # el (0,0) es el valor mas bajo de la matriz,
                                                                       # esta mas lejos del centro
    gauss = gauss.astype(np.uint8)
    print(gauss)
    return gauss


def normal_dist(x, deviation, mean=0):
    return (np.pi * deviation) * np.exp(-0.5 * ((x - mean) / deviation) ** 2)


def decimal_places(number):
    i = 0
    while number < 1:
        number *= 10
        i += 1
    return i
