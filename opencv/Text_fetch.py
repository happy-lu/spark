import cv2

import pytesseract as pt

if __name__ == '__main__':
    # 读取文件
    # imagePath = ""
    img = cv2.imread("E://ocr//db//112.jpg")

    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    dstHeight = int(height * 2)
    dstWidth = int(width * 2)

    dst = cv2.resize(img, (dstWidth, dstHeight))
    cv2.imshow('image', dst)

    # text = pt.image_to_string(img)
    # print(text)
    # print("----------------------------------")
    text = pt.image_to_string(dst)


    print(text)
    # detect(img)
