# -*- coding: utf-8 -*-
import cv2  as  cv
import numpy as  np
import os

file_dir_a = 'E://ocr//t4-1//'
file_dir_b = 'E://ocr//t4//'
savepath = '.'

all_file_name_a = os.listdir(file_dir_a)
all_file_name_b = os.listdir(file_dir_b)
image_all_a = []
image_all_b = []
for name in all_file_name_a:
    image_one = []
    image = cv.imread(file_dir_a + name, cv.IMREAD_GRAYSCALE)
    """arg是计算输入图片矩阵的特征值，通过对特征值的比较来实现图片的比对
    """
    # arg=np.linalg.eigvals(image)
    """arg是计算输入二值图片矩阵中1的个数，通过1的总数来实现图片的比对
    """
    arg = sum(image)
    image_one.append(name)
    image_one.append(arg)
    image_all_a.append(image_one)  # 将一个图片的信息写入
    print('读入a')
# np.save('img_a.npy',image_all_a)
for name in all_file_name_b:
    image_one = []
    image = cv.imread(file_dir_b + name, cv.IMREAD_GRAYSCALE)
    # arg=np.linalg.eigvals(image)
    arg = sum(image)
    image_one.append(name)
    image_one.append(arg)
    image_all_b.append(image_one)  # 将一个图片的信息写入
    print('读入b')
# np.save('img_b.npy',image_all_b)
print('开始比较')
result_all = []
for a in image_all_a:  # 比较小的
    result = []
    for b in image_all_b:
        # print sum(a[1]-b[1])
        if abs(sum(a[1] - b[1])) < 0.00001:
            result.append(a[0])
            result.append(b[0])
            result_all.append(result)
print('比较结束')
print(result_all)
