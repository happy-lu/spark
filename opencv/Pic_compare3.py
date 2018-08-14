import cv2
import numpy as np


def mathc_img(image, Target, value):
    img_rgb = cv2.imread(image)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # iw, ih = img_gray.shape[::-1]

    # img_gray = cv2.resize(img_gray, (800, 600), interpolation=cv2.INTER_CUBIC)

    template = cv2.imread(Target, 0)
    # cv2.imshow('ttt', template)
    # cv2.waitKey(0)

    w, h = template.shape[::-1]
    # template = cv2.resize(template, (int(w / size_change), int(h / size_change)), interpolation=cv2.INTER_CUBIC)

    # cv2.imshow('t1', template)
    # cv2.waitKey(0)
    #
    # cv2.imshow('t2', img_gray)
    # cv2.waitKey(0)

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = value
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (7, 249, 151), 2)
    cv2.imshow('Detected', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = ("E://ocr//np2.jpg")
    Target = ('E://ocr//np-1.jpg')
    value = 0.6
    mathc_img(image, Target, value)
