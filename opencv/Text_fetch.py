import cv2
import numpy as np
import pytesseract as pt

if __name__ == '__main__':
    # 读取文件
    # imagePath = ""
    img = cv2.imread("E://ocr//t8//30235297_res_res.jpg")
    gray = img
    #
    # gray = img
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # not_gray = cv2.bitwise_not(gray)
    # cv2.imshow('not', not_gray)
    #
    # row_nums = []
    # for i in range(len(gray)):
    #     if len(gray[i][gray[i] < 128]) > len(gray[i][gray[i] > 128]):
    #         row_nums.append(i)
    #
    #     # gray[i] = not_gray[i]
    # print(row_nums)
    #
    # c_points = 5
    # pre = 0
    # min = 0
    # if len(row_nums)>0:
    #     for i in row_nums:
    #         if i - pre >= c_points:
    #             min = i
    #         pre = i
    #     max = row_nums[-1] + 1
    #     gray[min:max, :] = not_gray[min:max, :]
    #     cv2.imshow('gray', gray)
    #
    # # text = pt.image_to_string(img)
    # # print(text)
    # # print("----------------------------------")

    # imgInfo = gray.shape
    # height = imgInfo[0]
    # width = imgInfo[1]
    # dstHeight = int(height / 3.15)
    # dstWidth = int(width / 3.15)
    #
    # dst = cv2.resize(gray, (dstWidth, dstHeight))
    # cv2.imshow('image', dst)


    text = pt.image_to_string(gray,lang="eng",config='--psm 4  --oem 1')
    print(text)

    # cv2.waitKey(0)
    # detect(img)
