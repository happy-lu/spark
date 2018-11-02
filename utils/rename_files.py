import os
import shutil
import time


def deal_folder():
    pic_folder = 'E:\\ocr\\test\\big\\big_image\\big_image'

    prefix = "cs_1_1_1_yehaobig_1010000001_0_98.254.1.151_db.ITMUSER_B_4.4.4.4_1521"
    new_folder = 'E:\\ocr\\test\\big\\big_image\\big_image\\' + prefix

    isExists = os.path.exists(new_folder)
    if not isExists:
        os.makedirs(new_folder)

    children = os.listdir(pic_folder)
    t = int(time.time() * 1000)
    for file_name in children:
        shutil.copy(os.path.join(pic_folder, file_name),
                    os.path.join(new_folder, prefix + "_" + str(t) + "_0_3696616.jpg"))
        t = t + 1000


if __name__ == '__main__':
    deal_folder()
