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
            if border[0] == 0 and border[1] == 255 and border[2] == 0:
                new_img[i][j] = border
            if corner[0] == 255 and corner[1] == 0 and corner[2] == 0:
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
            error_allowed = (0.75 - 0.5) / 2
            if 0.5 - error_allowed <= s < 0.5 + error_allowed:
                borders[i][j] = [0, 255, 0]
            if 0.5 + error_allowed <= s <= 0.75 + error_allowed:
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

def active_contours(img, left, up, right, down, avg_obj_color, epsilon, max_iterations):
    obj_mark = -3
    lin_mark = -1
    lout_mark = 1
    bg_mark = 3
    marks, lin, lout = initialize_marks(img, left, up, right, down, obj_mark, lin_mark, lout_mark, bg_mark)
    i = 0
    while not has_finished_active_contours(img, avg_obj_color, epsilon, lin, lout) and i < max_iterations:
        active_contours_iteration(img, avg_obj_color, epsilon, marks, lout, lin, lout_mark, lin_mark, obj_mark, bg_mark)
        i += 1
    new_img = copy(img)
    if len(img.shape) < 3:
        new_img = np.asarray(Image.fromarray(img).convert(mode='RGB'))
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            if (i, j) in lin:
                new_img[i][j] = [255, 0, 0]
            elif (i, j) in lout:
                new_img[i][j] = [0, 0, 255]

    return new_img.astype(uint8)


def active_contours_vid(img, marks, lin, lout, avg_obj_color, epsilon, max_iterations):
    obj_mark = -3
    lin_mark = -1
    lout_mark = 1
    bg_mark = 3
    i = 0
    while not has_finished_active_contours(img, avg_obj_color, epsilon, lin, lout) and i < max_iterations:
        print(i)
        active_contours_iteration(img, avg_obj_color, epsilon, marks, lout, lin, lout_mark, lin_mark, obj_mark, bg_mark)
        i += 1
    new_img = copy(img)
    if len(img.shape) < 3:
        new_img = np.asarray(Image.fromarray(img).convert(mode='RGB'))
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            if (i, j) in lin:
                new_img[i][j] = [255, 0, 0]
            elif (i, j) in lout:
                new_img[i][j] = [0, 0, 255]

    return new_img.astype(uint8), marks, lin, lout


def active_contours_iteration(img, avg_obj_color, epsilon, marks, lout, lin, lout_mark, lin_mark, obj_mark, bg_mark):
    expand_curve(img, avg_obj_color, epsilon, marks, lout, lin, lin_mark, lout_mark, bg_mark)
    aux_lin = list(lin)
    for pixel in aux_lin:
        if not is_touching_lout(pixel, marks, lout_mark):
            marks[pixel[0]][pixel[1]] = obj_mark
            lin.remove(tuple(pixel))
    contract_curve(img, avg_obj_color, epsilon, marks, lout, lin, lin_mark, lout_mark, obj_mark)
    aux_lout = list(lout)
    for pixel in aux_lout:
        if not is_touching_lin(pixel, marks, lin_mark):
            marks[pixel[0]][pixel[1]] = bg_mark
            lout.remove(tuple(pixel))


def fd(pixel_color, avg_obj_color, epsilon):
    if isinstance(pixel_color, int):
        if abs(pixel_color - avg_obj_color) < epsilon:
            return 1
        else:
            return -1
    else:
        if np.linalg.norm(pixel_color - avg_obj_color) < epsilon:
            return 1
        else:
            return -1


def is_touching_lout(pixel, marks, lout_mark):
    if pixel[0] - 1 >= 0:
        if marks[pixel[0] - 1][pixel[1]] == lout_mark:
            return True
    if pixel[0] + 1 < len(marks):
        if marks[pixel[0] + 1][pixel[1]] == lout_mark:
            return True
    if pixel[1] - 1 >= 0:
        if marks[pixel[0]][pixel[1] - 1] == lout_mark:
            return True
    if pixel[1] + 1 < len(marks[0]):
        if marks[pixel[0]][pixel[1] + 1] == lout_mark:
            return True
    return False

def is_touching_lin(pixel, marks, lin_mark):
    if pixel[0] - 1 >= 0:
        if marks[pixel[0] - 1][pixel[1]] == lin_mark:
            return True
    if pixel[0] + 1 < len(marks):
        if marks[pixel[0] + 1][pixel[1]] == lin_mark:
            return True
    if pixel[1] - 1 >= 0:
        if marks[pixel[0]][pixel[1] - 1] == lin_mark:
            return True
    if pixel[1] + 1 < len(marks[0]):
        if marks[pixel[0]][pixel[1] + 1] == lin_mark:
            return True

    return False


def initialize_marks(img, left, up, right, down, obj_mark=-3, lin_mark=-1, lout_mark=1, bg_mark=3):
    marks = np.zeros((img.shape[0], img.shape[1]), dtype=np.int8)
    lout = set()
    lin = set()
    for row in range(len(marks)):
        for col in range(len(marks[0])):
            if left < col < right and up < row < down:
                marks[row][col] = obj_mark
            elif (col == left or col == right) and up <= row <= down:
                marks[row][col] = lin_mark
                lin.add((row, col))
            elif (row == up or row == down) and left <= col <= right:
                marks[row][col] = lin_mark
                lin.add((row, col))
            elif (row == up - 1 or row == down + 1) and left <= col <= right:
                marks[row][col] = lout_mark
                lout.add((row, col))
            elif (col == left - 1 or col == right + 1) and up <= row <= down:
                marks[row][col] = lout_mark
                lout.add((row, col))
            else:
                marks[row][col] = bg_mark
    return marks, lin, lout


def expand_curve(img, avg_obj_color, epsilon, marks, lout, lin, lin_mark, lout_mark, bg_mark):
    aux_lout = list(lout)
    for pixel in aux_lout:
        if fd(img[pixel[0]][pixel[1]], avg_obj_color, epsilon) > 0:
            lout.remove(tuple(pixel))
            lin.add(tuple(pixel))
            marks[pixel[0]][pixel[1]] = lin_mark
            if pixel[0] + 1 < len(marks):
                if marks[pixel[0] + 1][pixel[1]] == bg_mark:
                    marks[pixel[0] + 1][pixel[1]] = lout_mark
                    lout.add((pixel[0] + 1, pixel[1]))
            if pixel[0] - 1 >= 0:
                if marks[pixel[0] - 1][pixel[1]] == bg_mark:
                    marks[pixel[0] - 1][pixel[1]] = lout_mark
                    lout.add((pixel[0] - 1, pixel[1]))
            if pixel[1] + 1 < len(marks[0]):
                if marks[pixel[0]][pixel[1] + 1] == bg_mark:
                    marks[pixel[0]][pixel[1] + 1] = lout_mark
                    lout.add((pixel[0], pixel[1] + 1))
            if pixel[1] - 1 >= 0:
                if marks[pixel[0]][pixel[1] - 1] == bg_mark:
                    marks[pixel[0]][pixel[1] - 1] = lout_mark
                    lout.add((pixel[0], pixel[1] - 1))


def contract_curve(img, avg_obj_color, epsilon, marks, lout, lin, lin_mark, lout_mark, obj_mark):
    aux_lin = list(lin)
    for pixel in aux_lin:
        if fd(img[pixel[0]][pixel[1]], avg_obj_color, epsilon) < 0:
            lin.remove(tuple(pixel))
            lout.add(tuple(pixel))
            marks[pixel[0]][pixel[1]] = lout_mark
            if pixel[0] + 1 < len(marks):
                if marks[pixel[0] + 1][pixel[1]] == obj_mark:
                    marks[pixel[0] + 1][pixel[1]] = lin_mark
                    lin.add((pixel[0] + 1, pixel[1]))
            if pixel[0] - 1 >= 0:
                if marks[pixel[0] - 1][pixel[1]] == obj_mark:
                    marks[pixel[0] - 1][pixel[1]] = lin_mark
                    lin.add((pixel[0] - 1, pixel[1]))
            if pixel[1] + 1 < len(marks[0]):
                if marks[pixel[0]][pixel[1] + 1] == obj_mark:
                    marks[pixel[0]][pixel[1] + 1] = lin_mark
                    lin.add((pixel[0], pixel[1] + 1))
            if pixel[1] - 1 >= 0:
                if marks[pixel[0]][pixel[1] - 1] == obj_mark:
                    marks[pixel[0]][pixel[1] - 1] = lin_mark
                    lin.add((pixel[0], pixel[1] - 1))


def has_finished_active_contours(img, avg_obj_color, epsilon, lin, lout):
    for pixel in lin:
        if fd(img[pixel[0]][pixel[1]], avg_obj_color, epsilon) < 0:
            return False
    for pixel in lout:
        if fd(img[pixel[0]][pixel[1]], avg_obj_color, epsilon) > 0:
            return False
    return True

