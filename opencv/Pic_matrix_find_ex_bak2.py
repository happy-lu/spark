from math import sqrt

import cv2
import os
import numpy as np


def get_to_right_top_length(point_x, point_y, right_vertex_x):
    x = abs(point_x - right_vertex_x)
    y = abs(point_y - 0)
    return sqrt(x * x + y * y)


def find_sql_area(pic_path, new_path):
    img = cv2.imread(pic_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grey", gray)
    ret, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary", binary)

    # blur and threshold the image
    blurred = cv2.blur(binary, (3, 3))
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    cv2.imshow("blurred", blurred)

    # # perform a series of erosions and dilations
    # blurred = cv2.erode(blurred, None, iterations=8)
    # blurred = cv2.dilate(blurred, None, iterations=8)
    # cv2.imshow("blurred2", blurred)

    r_pic, contours, hierarchy = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep_list = []
    for contour in contours:
        diff = contour.max(0) - contour.min(0)
        rec_width = diff[0][0]
        rec_height = diff[0][1]
        rec_area = rec_width * rec_height
        # 加入右上角的点
        if rec_width > rec_height and rec_area > 10000:
            length = get_to_right_top_length(contour.max(0)[0][0], contour.min(0)[0][1], img.shape[1])
            keep_list.append([length, contour])
    if len(keep_list) == 0:
        print("No suit area with picture %s, skip it" % (pic_path))
        return

    # # 取出最上方的矩阵，即与右上角最近的框框
    keep_list.sort()
    keep_contours = keep_list[0][1]

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
    cv2.imshow("img", crop_img)
    print("Successful deal with picture %s, the skr file is %s" % (pic_path, new_path))
    # cv2.imwrite(new_path, crop_img)

def show_one():
    find_sql_area('E://ocr//db//17.jpg', 'E://ocr//db//small//17.jpg')
    cv2.waitKey(0)


if __name__ == '__main__':
    # pic_folder = 'E://ocr//db_full//line'
    # new_folder = 'E://ocr//db_full//line//small'
    #
    # isExists = os.path.exists(new_folder)
    # if not isExists:
    #     os.makedirs(new_folder)
    #
    # children = os.listdir(pic_folder)
    # for file_name in children:
    #     file_path = os.path.join(pic_folder, file_name)
    #     if os.path.isfile(file_path):
    #         new_path = os.path.join(new_folder, file_name)
    #         find_sql_area(file_path, new_path)

    show_one()