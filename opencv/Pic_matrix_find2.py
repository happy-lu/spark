import cv2

img = cv2.imread('E://ocr//db//1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("grey", gray)
ret, binary = cv2.threshold(gray, 254, 1000, cv2.THRESH_BINARY)
cv2.imshow("binary", binary)





r_pic, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
# cv2.imshow("r_pic", r_pic)
cv2.imshow("img", img)
cv2.waitKey(0)