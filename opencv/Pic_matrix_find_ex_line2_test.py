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


def find_vertical_x_by_remove_lines(img):
    img_height = img.shape[0]
    tmp = np.where(img > 200, 1, 0)
    white_sum_array = tmp.sum(axis=0)
    with_line_x = np.where(white_sum_array >= img_height * 0.5)
    return with_line_x[0][0]


def find_vertical_lines(img, percent=0.5):
    img_height = img.shape[0]
    tmp = np.where(img > 200, 1, 0)
    white_sum_array = tmp.sum(axis=0)
    with_line_x = np.where(white_sum_array >= img_height * percent)[0]

    return with_line_x


def find_horizon_lines(img, percent=0.5):
    img_width = img.shape[1]
    tmp = np.where(img > 200, 1, 0)
    white_sum_array = tmp.sum(axis=1)
    with_line_x = np.where(white_sum_array >= img_width * 0.5)[0]

    return with_line_x


def find_sql_area(pic_path, new_path):
    img_full = cv2.imread(pic_path)
    img = img_full[50:-100, 100:-10]

    if len(img) == 0:
        print(pic_path + " is too small, skip it")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("grey", gray)
    ret, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_TOZERO_INV)
    cv2.imshow("binary", binary)

    ret, binary = cv2.threshold(binary, 65, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary2", binary)

    binary = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
    binary = cv2.convertScaleAbs(binary)
    cv2.imshow("binary3", binary)

    horizon_lines_y = find_horizon_lines(binary)
    vertical_lines_x = find_vertical_lines(binary)

    horizon_lines_y = sort_and_dedup(horizon_lines_y)
    vertical_lines_x = sort_and_dedup(vertical_lines_x)

    y_length = img.shape[0]
    x_length = img.shape[1]
    # for y in horizon_lines_y:
    #     cv2.line(img, (0, y), (x_length, y), (0, 0, 255), 1)
    #
    # for x in vertical_lines_x:
    #     cv2.line(img, (x, 0), (x, y_length), (0, 0, 255), 1)

    if len(vertical_lines_x) > 0:
        best_x = find_vertical_x_by_remove_lines(binary[vertical_lines_x[0]:, :])
        x_start = best_x - 1
    else:
        x_start = find_vertical_x_by_max_points(binary) - 3

    if len(horizon_lines_y) <= 2:
        print(pic_path + " has no suit area, skip it")
        return
    elif len(horizon_lines_y) == 3:
        # area between line 1 and 2
        y_start = horizon_lines_y[2]
        y_end = y_length
    else:
        # area between line 2 and 3
        y_start = horizon_lines_y[2]
        y_end = horizon_lines_y[3]

    crop_img = img[y_start:y_end, x_start:img.shape[1]]

    cv2.imshow("full_img", img)

    if len(crop_img) > 0:
        cv2.imshow("img", crop_img)
        cv2.imwrite(new_path, crop_img)
        print("Success save small pic to: " + new_path)


def show_one():
    find_sql_area('E://ocr//db//4.jpg', 'E://ocr//db//small//4.jpg')
    cv2.waitKey(0)


def deal_folder():
    pic_folder = 'E://ocr//db_full'
    new_folder = 'E://ocr//db_full//small2'

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
