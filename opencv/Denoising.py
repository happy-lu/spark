from math import sqrt

import cv2
import os
import numpy as np


# def remove_noise(pic_path):
#     img = cv2.imread(pic_path)
#     for each_col in img.T:
#         each_col.

def remove_noise(pic_path):
    img = cv2.imread(pic_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary", binary)

    cv2.imwrite(pic_path, binary)



def show_one():
    remove_noise('E://ocr//db//small//111.jpg')
    cv2.waitKey(0)


def deal_folder():
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
            remove_noise(file_path, new_path)


if __name__ == '__main__':
    show_one()
    # deal_folder()
