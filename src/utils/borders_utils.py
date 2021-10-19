import math

from PIL import Image
import numpy as np

from numpy import copy, int32, uint8

from src.utils.image_utils import synthesis, normalize
from src.utils.mask_utils import fill_outer_matrix, sobel_vertical_mask, sobel_horizontal_mask


def susan_filter_together(img, t):
    borders, corners = susan(img, t)
    new_img = copy(img)
    if len(img.shape) < 3:
        new_img = np.asarray(Image.fromarray(img).convert(mode='RGB'))
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            border = borders[i][j]
            corner = corners[i][j]
            if (border[0] == 0 and border[1] == 255 and border[2] == 0):
                new_img[i][j] = border
            if (corner[0] == 255 and corner[1] == 0 and corner[2] == 0):
                new_img[i][j] = corner

    return new_img.astype(uint8)
        

def susan_filter_apart(img, t):
    borders, corners = susan(img, t)
    new_img = copy(img)
    if len(img.shape) < 3:
        new_img = np.asarray(Image.fromarray(img).convert(mode='RGB'))
    border_img = copy(new_img)
    corner_img = copy(new_img)
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            border = borders[i][j]
            corner = corners[i][j]
            if (border[0] == 0 and border[1] == 255 and border[2] == 0):
                border_img[i][j] = border
            if (corner[0] == 255 and corner[1] == 0 and corner[2] == 0):
                corner_img[i][j] = corner


    return border_img.astype(uint8), corner_img.astype(uint8)


def susan(img, t):
    mask = susan_mask()
    borders = np.zeros((img.shape[0], img.shape[1], 3))
    corners = np.zeros((img.shape[0], img.shape[1], 3))
    aux_img = fill_outer_matrix(img, 7)
    for i in range(0, len(img)):
        end_aux_col = i + 7
        for j in range(0, len(img[0])):
            end_aux_row = j + 7
            # if len(img.shape) > 2:
            #     s = np.zeros(3)
            #     for k in range(0, len(img[0][0])):
            #         s[k] = calculate_s(aux_img[i: end_aux_col, j: end_aux_row, k], mask, t)
            #     avg_s = np.average(s)

            #     if(avg_s > 0.40 and avg_s < 0.60):
            #         borders[i][j] = [0, 255, 0]
            #     if(avg_s >= 0.60 and avg_s <= 0.80):
            #         corners[i][j] = [255, 0, 0]
            # else:
            s = calculate_s(aux_img[i: end_aux_col, j: end_aux_row], mask, t)
            if(s > 0.40 and s < 0.60):
                borders[i][j] = [0, 255, 0]
            if(s >= 0.60 and s <= 0.80):
                corners[i][j] = [255, 0, 0]
    return borders, corners


def susan_mask():
    mask = np.asarray(
        [
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ]
    )
    return mask


def calculate_s(img, mask, t):
    r0 = np.full(mask.shape, img[3][3]).astype(int32)
    aux = copy(img).astype(int32)
    re = np.subtract(aux, r0)
    subs = np.absolute(re)
    c = np.where(subs < t, 1, 0)
    applied = c * mask
    n = np.sum(applied)
    return 1 - (n / 37)


def canny(img, t1, t2):

    gy = sobel_vertical_mask(img)
    gy = gy.astype(np.int64)
    gy -= 127
    gx = sobel_horizontal_mask(img)
    gx = gx.astype(np.int64)
    gx -= 127
    g = synthesis(abs(gx), abs(gy))
    g = normalize(g, 0, 181)

    angles = angles_matrix(gx, gy)
    for i in range(len(g)):
        for j in range(len(g[0])):
            if g[i][j] != 0:
                if angles[i][j] == 0:
                    if (j-1 >= 0 and g[i][j-1] >= g[i][j]) or (j+1 < len(g[0]) and g[i][j+1] >= g[i][j]):
                        g[i][j] = 0
                elif angles[i][j] == 45:
                    if (i+1 < len(g) and j-1 >= 0 and g[i+1][j-1] >= g[i][j]) or \
                            (i-1 >= 0 and j+1 < len(g[0]) and g[i-1][j+1] >= g[i][j]):
                        g[i][j] = 0
                elif angles[i][j] == 90:
                    if (i+1 < len(g) and g[i+1][j] >= g[i][j]) or (i-1 >= 0 and g[i-1][j] >= g[i][j]):
                        g[i][j] = 0
                elif angles[i][j] == 135:
                    if (i-1 >= 0 and j-1 >= 0 and g[i-1][j-1] >= g[i][j]) or \
                            (i+1 < len(g) and j+1 < len(g[0]) and g[i+1][j+1] >= g[i][j]):
                        g[i][j] = 0

    ans = np.zeros(img.shape)
    for i in range(len(g)):
        for j in range(len(g[0])):
            if g[i][j] >= t2:
                ans[i][j] = 255

    for i in range(len(g)):
        for j in range(len(g[0])):
            if t1 < g[i][j] < t2:
                if (i-1 >= 0 and ans[i-1][j] == 255) or (i+1 < len(ans) and ans[i+1][j] == 255) or \
                        (j-1 >= 0 and ans[i][j-1] == 255) or (j+1 < len(ans[0]) and ans[i][j+1] == 255):
                    ans[i][j] = 255

    return ans


def angles_matrix(gx, gy):
    angles = []
    for i in range(len(gx)):
        angles.append([])
        for j in range(len(gx[0])):
            if gx[i][j] == 0:
                angles[i].append(90)
            else:
                angle = math.atan2(gy[i][j], gx[i][j])
                angle = angle_to_direction(angle)
                if angle == 135 or angle == 45:
                    print(i, j, gy[i][j], gx[i][j], gy[i][j] / gx[i][j], angle)
                angles[i].append(angle)
    return angles


def angle_to_direction(angle_in_rad):
    angle = math.degrees(angle_in_rad)
    if angle < 0:
        angle += 360
    if angle >= 180:
        angle -= 180
    interval = 22.5
    if angle < interval or 180 - interval <= angle < 180:
        return 0
    elif 45 - interval <= angle < 45 + interval:
        return 135
    elif 90 - interval <= angle < 90 + interval:
        return 90
    elif 135 - interval <= angle < 135 + interval:
        return 45

