import cv2
import numpy as np


def find_sql_area(file_name):
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("grey", gray)
    ret, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary", binary)
    r_pic, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    keep_list = []
    for contour in contours:
        diff = contour.max(0) - contour.min(0)
        rec_width = diff[0][0]
        rec_height = diff[0][1]
        rec_area = rec_width * rec_height
        if rec_width > rec_height and rec_area > 10000:
            keep_list.append([contour.min(0)[0][1], contour])
    if len(keep_list) == 0:
        print("No suit area")
        return
    # # 取出最上方的矩阵
    keep_list.sort()
    keep_contours = keep_list[0][1]
    kmax = keep_contours.max(0)
    kmin = keep_contours.min(0)
    pic_area = [[[kmin[0][0], kmin[0][1]]], [[kmax[0][0], kmin[0][1]]], [[kmin[0][0], kmax[0][1]]],
                [[kmax[0][0], kmax[0][1]]]]
    pic_nparea = np.array(pic_area)
    cv2.drawContours(img, pic_nparea, -1, (0, 0, 255), 3)
    # cv2.drawContours(img, keep_contours, -1, (0, 0, 255), 1)
    # cv2.drawContours(img, keep_area, -1, (0, 0, 255), 1)
    # cv2.imshow("r_pic", r_pic)
    cv2.imshow("img", img)


if __name__ == '__main__':
    find_sql_area('E://ocr//db//2.jpg')
    cv2.waitKey(0)
