from math import sqrt

import cv2
import os
import numpy as np


def get_to_right_top_length(point_x, point_y, right_vertex_x):
    x = abs(point_x - right_vertex_x)
    y = abs(point_y - 0)
    return sqrt(x * x + y * y)


def find_sql_area(pic_path, new_path):
    img_full = cv2.imread(pic_path)
    img = img_full[50:-100, :]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("grey", gray)
    ret, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_TOZERO_INV)

    ret, binary = cv2.threshold(binary, 50, 255, cv2.THRESH_BINARY)

    r_pic, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    keep_list = []
    for contour in contours:
        diff = contour.max(0) - contour.min(0)
        rec_width = diff[0][0]
        rec_height = diff[0][1]
        rec_area = rec_width * rec_height
        # 加入右上角的点
        if rec_width > rec_height and rec_area > 10000:
            keep_list.append(contour)
    if len(keep_list) == 0:
        print("No suit area")
        return

    # # 在所有轮廓中找到点
    keep_contours = np.concatenate(keep_list)

    rect = cv2.minAreaRect(keep_contours)
    box = np.int0(cv2.boxPoints(rect))

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    crop_img = img[y1:y1 + hight, x1:x1 + width]
    print("Successful deal with picture %s, the skr file is %s" % (pic_path, new_path))
    cv2.imwrite(new_path, crop_img)


if __name__ == '__main__':
    pic_folder = 'E://ocr//db_full'
    new_folder = 'E://ocr//db_full//small'

    isExists = os.path.exists(new_folder)
    if not isExists:
        os.makedirs(new_folder)

    children = os.listdir(pic_folder)
    for file_name in children:
        file_path = os.path.join(pic_folder, file_name)
        if os.path.isfile(file_path):
            new_path = os.path.join(new_folder, file_name)
            find_sql_area(file_path, new_path)
        # cv2.waitKey(0)
