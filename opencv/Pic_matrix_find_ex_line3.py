from math import sqrt

import cv2
import os
import numpy as np


def get_to_right_top_length(point_x, point_y, right_vertex_x):
    x = abs(point_x - right_vertex_x)
    y = abs(point_y - 0)
    return sqrt(x * x + y * y)


def sort_and_dedup(list, keep_small=True):
    list.sort()
    dedup_list = []
    temp = -100
    for value in list:
        if value - temp <= 20:
            if keep_small:
                continue
            else:
                dedup_list.pop()
                dedup_list.append(value)
                temp = value
        else:
            dedup_list.append(value)
            temp = value

    return dedup_list


def find_vertical_x_by_max_points(img):
    tmp = np.where(img > 200, 1, 0)
    white_sum_array = tmp.sum(axis=0)
    max_value = white_sum_array.max()
    return np.where(white_sum_array == max_value)[0][0]


def find_vertical_x_best_shift(img):
    img_height = img.shape[0]
    tmp = np.where(img > 200, 1, 0)
    white_sum_array = tmp.sum(axis=0)
    with_line_x = np.where(white_sum_array >= img_height * 0.5)

    tmp_value = 0
    max_value = 20
    for value in with_line_x[0]:
        if value < max_value:
            tmp_value = value
        else:
            break

    return tmp_value

def find_vertical_x_best_shift(img):
    img_height = img.shape[0]
    tmp = np.where(img > 200, 1, 0)
    white_sum_array = tmp.sum(axis=0)
    with_line_x = np.where(white_sum_array >= img_height * 0.5)

    tmp_value = 0
    max_value = 20
    for value in with_line_x[0]:
        if value < max_value:
            tmp_value = value
        else:
            break

    return tmp_value


def find_sql_area(pic_path, new_path):
    img_full = cv2.imread(pic_path)
    # img = img_full[50:-100, 100:-10]
    img = img_full

    if len(img) == 0:
        print(pic_path + " is too small, skip it")
        return


    cv2.imshow("", img)




def show_one():
    find_sql_area('E://ocr//db//211.jpg', 'E://ocr//db//small//211.jpg')
    cv2.waitKey(0)


def deal_folder():
    pic_folder = 'E://ocr//db'
    new_folder = 'E://ocr//db//small'

    isExists = os.path.exists(new_folder)
    if not isExists:
        os.makedirs(new_folder)

    children = os.listdir(pic_folder)
    for file_name in children:
        file_path = os.path.join(pic_folder, file_name)
        if os.path.isfile(file_path):
            new_path = os.path.join(new_folder, file_name)
            find_sql_area(file_path, new_path)


if __name__ == '__main__':
    show_one()
    # deal_folder()
