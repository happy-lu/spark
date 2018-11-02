import cv2
import numpy as np
import pytesseract as pt

if __name__ == '__main__':
    # 读取文件
    # imagePath = ""
    img = cv2.imread("E://ocr//1//222.jpg")
    gray = img


    imgInfo = gray.shape
    height = imgInfo[0]
    width = imgInfo[1]
    dstHeight = int(height * 3.125)
    dstWidth = int(width * 3.125)

    dst = cv2.resize(gray, (dstWidth, dstHeight))
    # cv2.waitKey(0)
    # detect(img)
