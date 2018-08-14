from math import sqrt

import cv2
import os
import numpy as np


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


def find__best_vertical_y(img):
    tmp = np.where(img > 200, 1, 0)
    white_sum_array = tmp.sum(axis=0)
    max_value = white_sum_array.max()
    return np.where(white_sum_array == max_value)[0][0]


def find_sql_area(pic_path, new_path):
    img_full = cv2.imread(pic_path)
    img = img_full[50:-100, 100:-10]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # get waterprint
    ret, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_TOZERO_INV)
    # make waterprint more clear
    ret, binary = cv2.threshold(binary, 65, 255, cv2.THRESH_BINARY)

    # get all lines
    lines = cv2.HoughLines(binary, 1, np.pi / 180, 250)

    if lines is None:
        print(pic_path + " has no lines, skip it")
        return

    horizon_lines_y = []
    vertical_lines_x = []

    for line in lines:
        for r, theta in line:
            # Stores the value of cos(theta) in a
            a = np.cos(theta)
            b = np.sin(theta)

            # vertical_lines
            if theta == 0:
                x = int(a * r + 1000 * (-b))
                vertical_lines_x.append(x)
            else:
                y = int(b * r + 1000 * a)
                horizon_lines_y.append(y)

    horizon_lines_y = sort_and_dedup(horizon_lines_y)
    vertical_lines_x = sort_and_dedup(vertical_lines_x, False)

    if len(vertical_lines_x) > 0:
        x_start = vertical_lines_x[0] - 1
    else:
        x_start = find__best_vertical_y(binary) - 3

    if len(horizon_lines_y) <= 2:
        print(pic_path + " has no suit area, skip it")
        return

    # area between line 1 and 2
    y_start = horizon_lines_y[1]
    y_end = horizon_lines_y[2]

    crop_img = img[y_start:y_end, x_start:img.shape[1]]
    if len(crop_img) > 0:
        # change to gray pic and remove the noise point, 160=255*0.63(sigmod)
        copy_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        copy_ret, copy_gray = cv2.threshold(copy_gray, 160, 255, cv2.THRESH_BINARY)
        cv2.imwrite(new_path, copy_gray)
        print("Success save small pic to: " + new_path)


def show_one():
    find_sql_area('E://ocr//db//4.jpg', 'E://ocr//db//small//4.jpg')
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
    # show_one()
    deal_folder()
