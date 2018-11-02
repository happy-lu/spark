import cv2
import os
import time
import numpy as np
from utils.logging_util import *
from utils.time_util import *

OPER_RESULT_SUCCESS = 0
OPER_RESULT_JUMPED = 1
OPER_RESULT_FAILED = 2


def sort_and_dedup(data_list, keep_small=True):
    data_list.sort()
    dedup_list = []
    temp = -100
    for value in data_list:
        if value - temp <= 20:
            if keep_small:
                continue
            else:
                dedup_list.pop()
                dedup_list.append(value)
                temp = value
        else:
            dedup_list.append(value)
            temp = value

    return dedup_list


def find_vertical_x_by_max_points(img):
    tmp = np.where(img > 200, 1, 0)
    white_sum_array = tmp.sum(axis=0)
    max_value = white_sum_array.max()
    return np.where(white_sum_array == max_value)[0][0]


def find_vertical_x_best_shift(img, max_shift=20, percent=0.5):
    img_height = img.shape[0]
    tmp = np.where(img > 200, 1, 0)
    white_sum_array = tmp.sum(axis=0)
    with_line_x = np.where(white_sum_array >= img_height * percent)

    tmp_value = 0
    for value in with_line_x[0]:
        if value < max_shift:
            tmp_value = value
        else:
            break

    return tmp_value


def find_vertical_lines(img, percent=0.5):
    img_height = img.shape[0]
    tmp = np.where(img > 200, 1, 0)
    white_sum_array = tmp.sum(axis=0)
    with_line_x = np.where(white_sum_array >= img_height * percent)[0]

    return with_line_x


def find_horizon_lines(img, percent=0.5):
    img_width = img.shape[1]
    tmp = np.where(img > 200, 1, 0)
    white_sum_array = tmp.sum(axis=1)
    with_line_x = np.where(white_sum_array >= img_width * percent)[0]

    return with_line_x


def find_sql_area(pic_path, new_path, show_debug_pic=False):
    """
    find the sql area and save it to new_path folder
    :param pic_path:
    :param new_path:
    :return: OPER_RESULT_xxx
    """
    img_full = cv2.imread(pic_path)

    y_top_shift = 50
    y_bottom_shift = -100
    x_left_shift = 100
    x_right_shift = -10
    vertical_x_max_shift = 20

    img = img_full[y_top_shift:y_bottom_shift, x_left_shift:x_right_shift]

    if len(img) == 0:
        logger.info(pic_path + " is too small, skip it")
        return OPER_RESULT_JUMPED

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # get waterlogger.info
    ret, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_TOZERO_INV)
    if show_debug_pic:
        cv2.imshow("binary", binary)

    # make waterprint more clear
    ret, binary = cv2.threshold(binary, 65, 255, cv2.THRESH_BINARY)
    if show_debug_pic:
        cv2.imshow("binary2", binary)

    binary = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
    binary = cv2.convertScaleAbs(binary)
    if show_debug_pic:
        cv2.imshow("binary3", binary)

    horizon_lines_y = find_horizon_lines(binary)
    horizon_lines_y = sort_and_dedup(horizon_lines_y)
    vertical_lines_x = find_vertical_lines(binary)
    vertical_lines_x = sort_and_dedup(vertical_lines_x)

    y_length = img.shape[0]
    x_length = img.shape[1]
    if show_debug_pic:
        for y in horizon_lines_y:
            cv2.line(img, (0, y), (x_length, y), (0, 0, 255), 1)
        for x in vertical_lines_x:
            cv2.line(img, (x, 0), (x, y_length), (0, 0, 255), 1)

    if len(horizon_lines_y) < 3:
        logger.info(pic_path + " has no suit area, skip it")
        return OPER_RESULT_JUMPED
    elif len(horizon_lines_y) == 3:
        # area between line 2 and 3
        y_start = horizon_lines_y[2]
        y_end = y_length
    else:
        # area between line 3 and 4
        y_start = horizon_lines_y[2]
        y_end = horizon_lines_y[3]

    if len(vertical_lines_x) > 0:
        # if snap area has many lines, choose the last one which x<20
        tmp = binary[y_start:y_end, vertical_lines_x[0]:]
        best_x = find_vertical_x_best_shift(tmp, vertical_x_max_shift)
        x_start = vertical_lines_x[0] + best_x - 1
    else:
        x_start = find_vertical_x_by_max_points(binary) - 3

    crop_img = img[y_start:y_end, x_start:img.shape[1]]
    if len(crop_img) > 0:
        # change to gray pic and remove the noise point, 160=255*0.63(sigmod)
        copy_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        copy_ret, copy_gray = cv2.threshold(copy_gray, 160, 255, cv2.THRESH_BINARY)
        cv2.imwrite(new_path, copy_gray)

        if show_debug_pic:
            cv2.imshow("origin", img)
            cv2.imshow("sql_area", crop_img)

        logger.info("Success save small pic to: " + new_path)

    return OPER_RESULT_SUCCESS


def show_one(src_file, dest_file):
    find_sql_area(src_file, dest_file, True)
    cv2.waitKey(0)


def deal_folder(pic_folder, new_folder):
    logger.info("The folder [%s] execute begin" % pic_folder)

    is_exists = os.path.exists(new_folder)
    if not is_exists:
        os.makedirs(new_folder)

    children = os.listdir(pic_folder)
    src_file_num = 0
    success = 0
    jumped_list = []
    exec_start_time = time.time()
    failed_list = []

    for file_name in children:
        file_path = os.path.join(pic_folder, file_name)
        if os.path.isfile(file_path):
            src_file_num += 1
            new_path = os.path.join(new_folder, file_name)

            try:
                result = find_sql_area(file_path, new_path)
                if result == OPER_RESULT_SUCCESS:
                    success += 1
                elif result == OPER_RESULT_JUMPED:
                    jumped_list.append(file_path)
                else:
                    failed_list.append(file_path)

            except Exception as err:
                failed_list.append(file_path)
                logger.error(file_path + " has occurred the below error:")
                logger.exception(err)

    time_taken = get_time_taken_str(time.time() - exec_start_time)
    logger.info("The folder [%s] execute finished" % pic_folder)
    if len(failed_list) == 0:
        logger.info("All files executed successful, total: %s, success: %d, jumped: %d, using time: %s" % (
            src_file_num, success, len(jumped_list), time_taken))
    else:
        logger.error(
            "Some files executed failed, total: %d, success: %d, jumped: %d, failed: %d, using time: %s" % (
                src_file_num, success, len(jumped_list), len(failed_list), time_taken))
        logger.error("Failed file list: %s" % str(failed_list))

    if len(jumped_list) > 0:
        logger.warning("Jumped file list: %s" % str(jumped_list))


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print("please add src/dest file or folder, like： python xxx.py /tmp/src.jpg /tmp/dst.jpg")
        exit()

    src_file = sys.argv[1]
    dest_file = sys.argv[2]

    logger = get_logger("sql_area_finder_log")
    try:
        if os.path.isfile(src_file):
            show_one(src_file, dest_file)
        else:
            deal_folder(src_file, dest_file)
    except Exception as err:
        logger.error("Occurred the below error:")
        logger.exception(err)
        raise err
