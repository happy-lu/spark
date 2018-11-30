import cv2
import numpy as np
import os, sys
import copy
import math

height_omit_all = 100#omit the first 100 lines in the picture ,that is the Control bar
time = 1
step = 10
#############################################################
#find aim line
def delete_wrong_point(delete_array,delete_list):
    for delete_num in (range(len(delete_list)-1,-1,-1)):
        del delete_array[delete_list[delete_num]]
        print (delete_array)

def countion_y_check(piexl_point, array):
    array.append(piexl_point)
    delete_point = []
    if (len(array) <= 3):
        return False
    elif(len(array) >= 4):
        for point in range(len(array)-1):
            if (point == (len(array)-1)):
                break
            elif((abs(array[point+1][0] - array[point][0]) == 1) and (abs(array[point+2][0] - array[point][0]) == 2) and (abs(array[point+3][0] - array[point][0]) == 3)):
                return True
            elif((abs(array[point+1][0] - array[point][0]) == 1) and (abs(array[point+2][0] - array[point][0]) == 2) and (abs(array[point+3][0] - array[point][0]) != 3)):
                delete_point.append(point)
                delete_point.append(point+1)
                delete_point.append(point+2)
                delete_wrong_point(array,delete_point)
                point  = 0
                return False
            elif ((abs(array[point + 1][0] - array[point][0]) == 1) and (abs(array[point + 2][0] - array[point][0]) != 2)):
                delete_point.append(point)
                delete_point.append(point + 1)
                delete_wrong_point(array, delete_point)
                point = 0
                return False
            elif(abs(array[point+1][0] - array[point][0]) != 1):
                delete_point.append(point)
                delete_wrong_point(array, delete_point)
                point = 0
                return False

def image_find_background_RGB(mat_image,mat_image_col,mat_image_row,height_omit):
    height  = mat_image_row
    width   = mat_image_col
    dst_mat_image = mat_image.copy()
    arrayl = []
    for piexl_height in range(height_omit, height):
        areaX = -1
        noiseCount = 0
        continuousCount = 0
        for piexl_width in range(0,width):
            actual_width = width - 1 - piexl_width
            piexl = dst_mat_image[piexl_height,actual_width]
            if ((piexl[0] >= 245 and piexl[1] >= 245 and piexl[2] >= 245) or (piexl[0] >= 80 and piexl[1] <= 25 and piexl[2] <= 25)):
                if(areaX <0):
                    areaX = actual_width
                else:
                    continuousCount += 1
                    if (continuousCount >=400):
                        piexl_point = [piexl_height,areaX]
                        if(countion_y_check(piexl_point, arrayl)):
                            return arrayl[0]
                        break
            else:
                noiseCount += 1
                if(noiseCount>=150):
                    areaX = -1
                    noiseCount = 0

def horizontal_detect(line,start_point,suitable_h_line):
    if (len(suitable_h_line) == 0):
        suitable_h_line = line
    else:
        if(abs(suitable_h_line[0] - start_point[0]) >abs(line[0] - start_point[0])):
            suitable_h_line =line

def vertical_detect(line,start_point,suitable_v_line):
    print (type(suitable_v_line))
    if (len(suitable_v_line) == 0):
        suitable_v_line = line
    else:
        if(abs(suitable_v_line[1] - start_point[1]) >abs(line[1] - start_point[1])):
            suitable_v_line = line

def detect_suitable_line(lines, check_point):
    start_point = [-1,-1]
    start_point[0] = check_point[1]
    start_point[1] = check_point[0]
    suit_h_line = []
    suit_v_line = []
    for detect_line in lines:
        x1 = detect_line[0]
        x2 = detect_line[2]
        y1 = detect_line[1]
        y2 = detect_line[3]
        if(x1 == x2):
            if (len(suit_h_line) == 0):
                suit_h_line = detect_line
            else:
                if (abs(suit_h_line[0] - start_point[0]) > abs(detect_line[0] - start_point[0])):
                    suit_h_line = detect_line
        if(y1 == y2):
            if (len(suit_v_line) == 0):
                suit_v_line = detect_line
            else:
                if (abs(suit_v_line[1] - start_point[1]) > abs(detect_line[1] - start_point[1])):
                    suit_v_line = detect_line
    return suit_h_line,suit_v_line

def search_horizontal_right_line(img_rgb,img_bin,line_v,start_piexl,info_img):
    x1 = line_v[0]
    y1 = line_v[1]
    x2 = start_piexl[1]
    y2 = line_v[3]
    rows = info_img[0]
    cols = info_img[0]
    for piexl_width in range(x2, x1,-1):
        noiseCount = 0
        continuousCount = 0
        noise_continuous = 0
        for piexl_height in range(y1,rows):
            piexl_bin = img_bin[piexl_height, piexl_width]
            piexl_rgb = img_rgb[piexl_height, piexl_width]
            if ((piexl_bin >230 ) or (piexl_rgb[0] >= 70 and piexl_rgb[1] <= 40 and piexl_rgb[2] <= 40)or(piexl_rgb[0] <= 10 and piexl_rgb[1] <= 10 and piexl_rgb[2] <= 10)):
                continuousCount += 1
                noise_continuous = 0
            else:
                noiseCount += 1
                noise_continuous += 1
                if(noiseCount>10):
                    break
                if ((noise_continuous >= 4) and (continuousCount > 20)):
                    return [piexl_width, piexl_height-4]

def search_vertical_down_line(img_rgb, img_bin, right_h_point, info_img):
    #right_h_point :::: [piexl_width, piexl_height-4]
    for piexl_height in range(right_h_point[1],0,-1):
        noiseCount = 0
        continuousCount = 0
        noise_continuous = 0
        for piexl_width in range(right_h_point[0],0,-1):
            piexl_bin = img_bin[piexl_height, piexl_width]
            piexl_rgb = img_rgb[piexl_height, piexl_width]
            if ((piexl_bin > 230) or (piexl_rgb[0] >= 70 and piexl_rgb[1] <= 40 and piexl_rgb[2] <= 40) or (
                    piexl_rgb[0] <= 10 and piexl_rgb[1] <= 10 and piexl_rgb[2] <= 10)):
                continuousCount += 1
                noise_continuous = 0
            else:
                noiseCount += 1
                noise_continuous += 1
                if (noiseCount > 10):
                    break
                if ((noise_continuous >= 2) and (continuousCount > 120)):
                    return [piexl_width-3, piexl_height]

def search_horizontal_left_line(img_rgb, img_bin, left_v_point, info_img):
    # left_v_point  [piexl_width, piexl_height-4]
    #    info_img   [rows, cols]
    for piexl_width in range(left_v_point[0], info_img[1]):
        noiseCount = 0
        continuousCount = 0
        noise_continuous = 0
        for piexl_height in range(left_v_point[1],0,-1):
            piexl_bin = img_bin[piexl_height, piexl_width]
            piexl_rgb = img_rgb[piexl_height, piexl_width]
            if ((piexl_bin > 230) or (piexl_rgb[0] >= 70 and piexl_rgb[1] <= 40 and piexl_rgb[2] <= 40) or (
                    piexl_rgb[0] <= 10 and piexl_rgb[1] <= 10 and piexl_rgb[2] <= 10)):
                continuousCount += 1
                noise_continuous = 0
            else:
                noiseCount += 1
                noise_continuous += 1
                if (noiseCount > 10):
                    break
                if ((noise_continuous >= 4) and (continuousCount > 20)):
                    return [piexl_width, piexl_height+5]

def precise_img_height(image):
    rows = image.shape[0] #height
    cols = image.shape[1] #width\
    ret_height = 0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    wb, image = cv2.threshold(image, 233, 255, cv2.THRESH_BINARY)
    for piexl_height in range(rows-2,rows-30,-1):
        noiseCount = 0
        continuousCount = 0
        for piexl_width in range(0, cols):
            piexl = image[piexl_height, piexl_width]
            if (piexl>= 240):
                continuousCount += 1
            else:
                noiseCount += 1
            if (noiseCount > 20):
                break
            elif (continuousCount > (cols-30)):
                ret_height = rows - piexl_height
                return ret_height
    if ret_height == 0:
        return 0

def precise_img_width(image):
    rows = image.shape[0]  # height
    cols = image.shape[1]  # width\
    ret_width = 0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    wb, image = cv2.threshold(image, 233, 255, cv2.THRESH_BINARY)
    for piexl_width in range(2,20):
        noiseCount = 0
        continuousCount = 0
        noise_continuous = 0
        for piexl_height in range(0, rows):
            piexl = image[piexl_height, piexl_width]
            if (piexl >= 240 ):
                continuousCount += 1
            else:
                noiseCount += 1
            if (noiseCount > 20):
                break
            elif (continuousCount > rows-30):
                ret_width = piexl_width
                return ret_width
    if (ret_width == 0):
        return 0

def all_bg_search(image):
    rows = image.shape[0]
    cols = image.shape[1]
    info_img = [rows, cols]
    dst_mat_image = np.zeros(image.shape, np.uint8)
    start_piexl = image_find_background_RGB(image, cols, rows, height_omit_all)
    print(start_piexl)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    wb, wb1 = cv2.threshold(gray, 233, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 50, 155)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=200, maxLineGap=3)
    print(lines[0])
    lines1 = lines[0]
    suitable_horizontal_line, suitable_vertical_line = detect_suitable_line(lines1, start_piexl)
    print("suitable_horizontal_line, suitable_vertical_line", suitable_horizontal_line, suitable_vertical_line)
    h = search_horizontal_right_line(image,wb1,suitable_vertical_line, start_piexl,info_img)
    v = search_vertical_down_line(image,wb1, h, info_img)
    l = search_horizontal_left_line(image, wb1, v, info_img)
    right_up = [h[0],suitable_vertical_line[3]]
    right_down = [h[0], v[1]]
    left_down = [l[0],v[1]]
    #cv2.rectangle(image, (l[0],v[1]), (h[0], suitable_vertical_line[3]), (0, 0, 255), 1)
    add_width = 0
    min_height = 0
    tmp = image[right_up[1]:left_down[1], left_down[0]:right_up[0]]
    min_height = precise_img_height(tmp)
    tmp = image[right_up[1] + 1:left_down[1]-min_height, left_down[0]:right_up[0] - 1]
    add_width = precise_img_width(tmp)
    print ("[add_width, min_height]",[add_width, min_height])
    cv2.imwrite("Image.jpg", image[right_up[1]+1:left_down[1]-min_height , left_down[0]+ add_width:right_up[0]-1])
    return image[right_up[1]+1:left_down[1]-min_height , left_down[0]+ add_width:right_up[0]-1]
###########################################################
#background change
#start func is "blue_bg_change2white"  input img:BGR image; img_gray:GRAY image
#EXP:img = blue_bg_change2white(img,img_grey)
def find_bg_selectarea(mask,img_grey):
    kernel2 = np.ones((5, 5), np.uint8)
    kernel3 = np.ones((7, 7), np.uint8)
    kernel_mor = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_mor)
    mask = cv2.dilate(mask, kernel2, iterations=1)
    mask = cv2.dilate(mask, kernel3, iterations=1)
    mask = cv2.erode(mask, kernel3, iterations=1)
    mask = cv2.erode(mask, kernel2, iterations=1)
    kernel_kor = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_kor)
    kernel4 = np.ones((3, 1), np.uint8)
    mask = cv2.dilate(mask, kernel4, iterations=1)
    select_area = 255-(((mask / 255) * img_grey) + ((255 - mask)))
    return select_area

def find_bg_selectarea_2(mask,img_grey):
    kernel2 = np.ones((3, 1), np.uint8)
    kernel3 = np.ones((5, 3), np.uint8)
    kernel_mor = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_mor)
    mask = cv2.dilate(mask, kernel2, iterations=1)
    mask = cv2.dilate(mask, kernel3, iterations=1)
    mask = cv2.erode(mask, kernel3, iterations=1)
    mask = cv2.erode(mask, kernel2, iterations=1)
    kernel_kor = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_kor)
    kernel4 = np.ones((1, 2), np.uint8)
    mask = cv2.dilate(mask, kernel4, iterations=1)
    select_area = 255-(((mask / 255) * img_grey) + ((255 - mask)))
    return select_area

def blue_bg_change2white(img,img_grey):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 43, 46])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    select_area = find_bg_selectarea(mask, img_grey)
    cv2.imshow("select_area", select_area)
    select_area = find_bg_selectarea_2(select_area, mask)
    cv2.imshow("select_area2", select_area)
    ret, select_area = cv2.threshold(select_area, 127, 255, cv2.THRESH_BINARY)
    show = cv2.split(img)
    show[2] = (255-select_area)/255*show[2]
    show[1] = (255-select_area)/255*show[1]
    show[0] = (255-select_area)/255*show[0]
    select_area_finally = 255 - (((select_area / 255) * img_grey) + ((255 - select_area)))
    ret, select_area_finally = cv2.threshold(select_area_finally, 180, 255, cv2.THRESH_BINARY)
    show[2] = show[2]+select_area_finally
    show[1] = show[1]+select_area_finally
    show[0] = show[0]+select_area_finally
    cv2.merge(show, img)
    return img
####################################################
#noise clear
#start func is "contrasStretch"  input srcImage:gray image; step:int(Row)
#gray  = contrasStretch(img1,14)
def find_max_T(array):
    length = len(array)
    max = 0
    max_targ = -1
    for i in range(length):
        if(max <= array[i]):
            max = array[i]
            max_targ = i
    if(-1<max_targ):
        return max_targ
    else:
        return -1

def OTSU_np(calu_image, row_start, col_start, row_span, col_span, T_step):
    T_array = []
    for T in range(170,236,6):
        bg_cal = 1.0
        bg_pix_all = 1.0
        fg_cal = 1.0
        fg_pix_all = 1.0
        tmp_lar_T = (calu_image[row_start:(row_start + row_span), col_start:(col_start + col_span)] >= T)
        tmp_sma_T = np.ones((row_span,col_span)) - tmp_lar_T
        fg_cal = sum(sum(tmp_lar_T)) + 1.0
        fg_pix_all = sum(sum(calu_image[row_start:(row_start+row_span),col_start:(col_start+col_span)]*
                             tmp_lar_T))+1.0
        bg_cal = sum(sum(calu_image[row_start:(row_start + row_span), col_start:(col_start + col_span)] < T)) + 1.0
        bg_pix_all = sum(sum(calu_image[row_start:(row_start + row_span), col_start:(col_start + col_span)] *
                             tmp_sma_T)) + 1.0

        Wf = fg_cal / (bg_cal+fg_cal)
        Wb = 1 - Wf
        Uf = fg_pix_all / fg_cal
        Ub = bg_pix_all / bg_cal
        T_array.append(Wf*Wb*(Ub-Uf)*(Ub-Uf))
    finally_T = T_step * find_max_T(T_array)+170
    print(finally_T)
    for j in range(row_start, row_start+row_span):
        for i in range(col_start, (col_start + col_span)):
            if(calu_image[j][i] >= finally_T):
                calu_image[j][i] = 255
            else:
                calu_image[j][i] = 0
    return calu_image

def OTSU(calu_image, row_start, col_start, row_span, col_span, T_step):
    T_array = []
    for T in range(170, 236, 6):
        bg_cal = 1.0
        bg_pix_all = 1.0
        fg_cal = 1.0
        fg_pix_all = 1.0

        for j in range(row_start, (row_start + row_span)):
            for i in range(col_start, (col_start + col_span)):
                if (calu_image[j][i] >= T):
                    fg_cal += 1
                    fg_pix_all += calu_image[j][i]
                else:
                    bg_cal += 1
                    bg_pix_all += calu_image[j][i]
        Wf = fg_cal / (bg_cal + fg_cal)
        Wb = 1 - Wf
        Uf = fg_pix_all / fg_cal
        Ub = bg_pix_all / bg_cal
        T_array.append(Wf * Wb * (Ub - Uf) * (Ub - Uf))
    finally_T = T_step * find_max_T(T_array) + 170
    print(finally_T)
    for j in range(row_start, row_start + row_span):
        for i in range(col_start, (col_start + col_span)):
            if (calu_image[j][i] >= finally_T):
                calu_image[j][i] = 255
            else:
                calu_image[j][i] = 0
    return calu_image

def contrasStretch(srcImage, step):
    resultImage = srcImage.copy()
    nRows = resultImage.shape[0]
    nCols = resultImage.shape[1]
    pixMax = 255
    pixMin = 0
    s_nCols = nCols/20 + 1
    for i in range(0, s_nCols):
        length_col = 20
        if(i == (s_nCols -1)):
            length_col = nCols - (s_nCols - 1)*20
        for j in range(0, nRows):
            if(j>= (nRows - step)):
                resultImage = OTSU_np(resultImage, j , i*20, (nRows - j), length_col, 6)
            else:
                resultImage = OTSU_np(resultImage, j, i * 20, 14, length_col, 6)
    return resultImage
####################################################
#delete miss part line
#start func is "all_vs"  input ori_img:binary image
#EXP:img = all_vs(img_BIN)
def judge_whiteline(line, nCols):
    white = 0
    black = 0
    for i in range(nCols-150):
        if(255 == line[i]):
            white += 1
        if(0 == line[i]):
            black += 1
    if(white >= (nCols-160)):
        return 1
    elif (black > 20):
        return -1
    else:
        return 0

def check_missingline(up_array, down_array, up_len, down_len):
    if(up_array[0] > down_array[0]):
        return 1
    if(up_array[up_len] > down_array[down_len]):
        return -1

def find_vs(resultImage, ori_img):
    nRows = resultImage.shape[0]
    nCols = resultImage.shape[1]
    line_before = 0
    queue_up = []
    queue_down = []
    queue_down_int = 0
    queue_up_int = 0
    for j in range(nRows):
        line = resultImage[j]
        line_find = judge_whiteline(line, nCols)
        if((line_before < 0) and (line_find > 0)):
            queue_down.append(j)
            queue_down_int += 1
        if((line_before > 0) and (line_find < 0)):
            queue_up.append(j)
            queue_up_int += 1
        line_before = line_find
    judge_num = check_missingline(queue_up, queue_down, queue_up_int - 1, queue_down_int - 1)
    print (judge_num)
    divi_img = ori_img.copy()
    if(judge_num < 0):
        for j in range(queue_up[(queue_up_int -1)], nRows):
            for i in range(nCols):
                print (j,i)
                divi_img[j][i] = 255
    elif (judge_num > 0):
        for j in range(0,queue_down[(queue_down_int -1)]):
            for i in range(nCols):
                divi_img[j][i] = 255
    else:
        iv = 0
    print (queue_down,queue_up)
    for j in range(queue_down_int):
        if(not((j+1))%1):
            for i in range(nCols):
                divi_img[(queue_down[j])][i] = 0

    return divi_img

def all_vs(ori_img):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    mask2 = cv2.erode(ori_img, element, iterations = 1)
    mask2 = cv2.erode(mask2, element1, iterations = 1)
    output = find_vs(mask2, ori_img)
    return output
####################################################
if __name__ == "__main__":
    image = cv2.imread("316.jpg", -1)
    #print "X",start_piexl
    #print "start_piexl",image[start_piexl[0],start_piexl[1]]
    '''
    if (v):
        for j in range(9):
            for i in range(9):
                image[v[1] - i, v[0] - j] = [0, 0, 252]
    '''

    new_image = all_bg_search(image)
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    img = blue_bg_change2white(new_image, gray)
    # cv2.imshow("select_area3", img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # size = (int(new_image.shape[1] * 2.2), int(new_image.shape[0] * 2.2))
    # image1 = cv2.resize(gray, size, interpolation=cv2.INTER_LINEAR)
    # #img = contrasStretch(gray,14)
    # cv2.imwrite("fin.jpg", image1)
    # '''
    # image1 = cv2.imread("fin.jpg", -1)
    # size = (int(image1.shape[1] * 2.2), int(image1.shape[0] * 2.2))
    # image1 = cv2.resize(image1, size, interpolation=cv2.INTER_NEAREST )
    # wb, image1 = cv2.threshold(image1, 233, 255, cv2.THRESH_BINARY)
    # img = all_vs(image1)
    #
    # cv2.imshow("Image", img)
    #
    #
    # cv2.imwrite("fin1.jpg", img)
    # '''
    cv2.waitKey(0)
