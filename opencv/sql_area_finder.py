import cv2
import os
import time
import numpy as np
from utils.logging_util import *
from utils.time_util import *

OPER_RESULT_SUCCESS = 0
OPER_RESULT_JUMPED = 1
OPER_RESULT_FAILED = 2


def sort_and_dedup(list, keep_small=True):
    list.sort()
    dedup_list = []
    temp = -100
    for value in list:
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


def find_vertical_x_best_shift(img, max_shift=20):
    img_height = img.shape[0]
    tmp = np.where(img > 200, 1, 0)
    white_sum_array = tmp.sum(axis=0)
    with_line_x = np.where(white_sum_array >= img_height * 0.5)

    tmp_value = 0
    for value in with_line_x[0]:
        if value < max_shift:
            tmp_value = value
        else:
            break

    return tmp_value


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
    ret, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_TOZERO_INV)
    if show_debug_pic:
        cv2.imshow("binary", binary)

    # make waterprint more clear
    ret, binary = cv2.threshold(binary, 65, 255, cv2.THRESH_BINARY)
    if show_debug_pic:
        cv2.imshow("binary2", binary)

    # get all lines
    lines = cv2.HoughLines(binary, 1, np.pi / 180, 250)

    if lines is None:
        logger.info(pic_path + " has no lines, skip it")
        return OPER_RESULT_JUMPED

    horizon_lines_y = []
    vertical_lines_x = []

    for line in lines:
        for r, theta in line:
            # Stores the value of cos(theta) in a
            a = np.cos(theta)
            b = np.sin(theta)

            # vertical_lines
            if theta == 0:
                x = int(a * r + 1000 * (-b))
                vertical_lines_x.append(x)
            else:
                y = int(b * r + 1000 * a)
                horizon_lines_y.append(y)

    horizon_lines_y = sort_and_dedup(horizon_lines_y)
    vertical_lines_x = sort_and_dedup(vertical_lines_x)

    if len(vertical_lines_x) > 0:
        # if snap area has many lines, choose the last one which x<20
        tmp = binary[:, vertical_lines_x[0]:]
        best_x = find_vertical_x_best_shift(tmp, vertical_x_max_shift)
        x_start = vertical_lines_x[0] + best_x - 1
    else:
        x_start = find_vertical_x_by_max_points(binary) - 3

    if len(horizon_lines_y) <= 2:
        logger.info(pic_path + " has no suit area, skip it")
        return OPER_RESULT_JUMPED

    # area between line 2 and 3
    y_start = horizon_lines_y[1]
    y_end = horizon_lines_y[2]

    crop_img = img[y_start:y_end, x_start:img.shape[1]]
    if len(crop_img) > 0:
        # change to gray pic and remove the noise point, 160=255*0.63(sigmod)
        copy_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        copy_ret, copy_gray = cv2.threshold(copy_gray, 160, 255, cv2.THRESH_BINARY)
        cv2.imwrite(new_path, copy_gray)

        if show_debug_pic:
            cv2.imshow("origin", img)
            cv2.imshow("sql", crop_img)

        logger.info("Success save small pic to: " + new_path)

    return OPER_RESULT_SUCCESS


def show_one(src_file, dest_file):
    find_sql_area(src_file, dest_file, True)
    cv2.waitKey(0)


def deal_folder(pic_folder, new_folder):
    is_exists = os.path.exists(new_folder)
    if not is_exists:
        os.makedirs(new_folder)

    children = os.listdir(pic_folder)
    src_file_num = 0
    success = 0
    jumped = 0
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
                    jumped += 1
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
            src_file_num, success, jumped, time_taken))
    else:
        logger.warning(
            "Some files executed failed, total: %d, success: %d, jumped: %d, failed: %d, using time: %s" % (
                src_file_num, success, jumped, len(failed_list), time_taken))
        logger.warning("Failed file list: %s" % str(failed_list))


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print("please add src_file and dest_file path, like： python xxx.py /tmp/src.jpg /tmp/dst.jpg")
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
        logger.error("Command has occurred the below error:")
        logger.exception(err)
        raise err
