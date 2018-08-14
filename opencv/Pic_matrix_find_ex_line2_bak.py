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


def find_sql_area(pic_path, new_path):
    img_full = cv2.imread(pic_path)
    img = img_full[50:-100, 100:-10]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("grey", gray)
    ret, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_TOZERO_INV)
    cv2.imshow("binary", binary)

    ret, binary = cv2.threshold(binary, 65, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary2", binary)

    lines = cv2.HoughLines(binary, 1, np.pi / 180, 250)

    keep_list = []
    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    horizon_lines_y = []
    vertical_lines_x = []

    if lines is None:
        print(pic_path + " has no lines, skip it")
        return

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

            # x0 stores the value rcos(theta)
            x0 = a * r
            y0 = b * r

            y1 = int(y0 + 1000 * (a))
            y2 = int(y0 - 1000 * (a))

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000 * (-b))
            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000 * (-b))

            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            # (0,0,255) denotes the colour of the line to be
            # drawn. In this case, it is red.
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    horizon_lines_y = sort_and_dedup(horizon_lines_y)
    vertical_lines_x = sort_and_dedup(vertical_lines_x)


    x_start = find_vertical_x_by_max_points(binary) - 1

    img_x_length = img.shape[1]
    y_start = 0
    y_end = 0
    if len(horizon_lines_y) <= 2:
        print(pic_path + " has no suit area, skip it")
        return
    elif len(horizon_lines_y) <= 4:
        # area between line 1 and 2
        y_start = horizon_lines_y[1]
        y_end = horizon_lines_y[2]
    else:
        # area between line 2 and 3
        y_start = horizon_lines_y[1]
        y_end = horizon_lines_y[2]

    crop_img = img[y_start:y_end, x_start:img.shape[1]]

    cv2.imshow("full_img", img)

    if len(crop_img) > 0:
        cv2.imshow("img", crop_img)
        cv2.imwrite(new_path, crop_img)
        print("Success save small pic to: " + new_path)


def show_one():
    find_sql_area('E://ocr//db//9.jpg', 'E://ocr//db//small//9.jpg')
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
