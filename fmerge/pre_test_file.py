import os
from utils.logging_util import *
import time


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


def create_test_file(full_text):
    father_path = os.path.abspath(os.path.dirname(full_text) + os.path.sep + ".")
    father_path = os.path.join(father_path, full_text[0:-4] + "_sub")
    is_exists = os.path.exists(father_path)
    if is_exists:
        return
        # del_file(father_path)
        # os.removedirs(father_path)

    os.makedirs(father_path)

    full_str = ""
    with open(full_text, "r") as file:
        lines = file.readlines()
        full_str = "".join(lines)

    print("full text:\n" + full_str)
    file_num = 5
    step = int(len(full_str) / file_num)
    cur_time = int(time.time() / 1000) * 1000

    for i in range(file_num):
        i += 1
        if i < file_num:
            cur_str = full_str[0:i * step]
        else:
            cur_str = full_str

        file_name = str(int(cur_time + i * 1000))
        file_name = os.path.join(father_path, file_name)
        f1 = open(file_name + '.txt', 'w')
        f1.write(cur_str)
        f1.close()

    print("done")


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print("please add src folder, like： python xxx.py /tmp/")
    #     exit()
    # src_file = sys.argv[1]

    src_folder = 'E://ocr//text//sql'
    children = os.listdir(src_folder)

    for file_name in children:
        try:
            new_file = os.path.join(src_folder, file_name)
            if os.path.isfile(new_file):
                create_test_file(new_file)
        except Exception as err:
            print("Command has occurred the below error:")
            print(err)
            raise err
