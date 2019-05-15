import os
import re
import datetime
from copy import deepcopy
from utils.logging_util import *


class FileMerger(object):
    def __init__(self, conf):
        self.config = conf

    def split_by_keywords(self, full_str):
        str_list = full_str.split(";")
        str_len = len(str_list)
        result_list = []
        for i, str in enumerate(str_list):
            str = str.replace("\r", " ").replace("\n", " ").replace("\t", " ")
            str = re.sub(re.compile(" +"), " ", str).strip()

            if i < str_len - 1:
                str += ";"
            result_list.append(str)

        return result_list

    def get_new_items(self, old_list, new_list):
        r_list = [val for val in new_list if val not in old_list]
        return r_list

    def get_time_by_filename(self, name):
        return int(name[:-4])

    def merge_files(self, input_str_dict, logger):
        # logger.info("input file num:" + str(len(str_list)))
        logger.debug("input file:" + str(input_str_dict))
        input_str_list = sorted(input_str_dict.items(), key=lambda x: x[0], reverse=False)

        confirm_dict = {}
        unconfirm_tuple = []
        recent_items_count = int(self.config['keep_items_count'])
        tail_index = 100

        for index, (k, v) in enumerate(input_str_list):
            value_list = self.split_by_keywords(v)

            # don't have confirmed contents
            if len(value_list) == 1:
                unconfirm_tuple = (k, value_list[0]) if value_list[0] != "" else unconfirm_tuple
                continue

            # if confirm_dict is empty, add all confirmed rows to it
            if len(confirm_dict) == 0:
                values_len = len(value_list)
                for i, each_value in enumerate(value_list):
                    if i < values_len - 1:
                        confirm_dict[k, i] = each_value
                    else:
                        unconfirm_tuple = (k, each_value) if each_value != "" else unconfirm_tuple
                continue

            #  insert skr diff cmd to confirm_dict
            old_list = list(confirm_dict.values())
            comp_count = recent_items_count if len(old_list) > recent_items_count else len(old_list)
            comp_list = old_list[-comp_count:]
            new_confirmed_list = self.get_new_items(comp_list, value_list[:-1])
            # have skr cmd
            if len(new_confirmed_list) > 0:
                for i, value in enumerate(new_confirmed_list):
                    # 1st confirmed unconfirm_tuple. because gui may delete all commands then add skr commands
                    if unconfirm_tuple:
                        confirm_dict[unconfirm_tuple[0], tail_index] = unconfirm_tuple[1]
                        unconfirm_tuple = None

                    confirm_dict[k, i] = value
            # add last command to unconfirm_tuple
            unconfirm_tuple = (k, value_list[-1]) if value_list[-1] != "" else unconfirm_tuple

        if unconfirm_tuple:
            confirm_dict[unconfirm_tuple[0], tail_index] = unconfirm_tuple[1]

        logger.debug("merge_files successful, result:")
        for k, v in confirm_dict.items():
            logger.debug(str(k) + ":" + v)

        return confirm_dict

    def parse_to_json(self, cmd_dict, info_dict, logger):
        result = {"index": "sql_analyze", "data": []}
        number = 0
        for (k1, k2), v in cmd_dict.items():
            c_data = deepcopy(info_dict)
            c_data["command"] = v
            c_data["number"] = number
            c_data["execute_time"] = k1
            c_data["path"] = "path"
            result["data"].append(c_data)
            number += 1

        logger.debug("parse_to_json successful, str:" + str(result))
        return result

    def read_all_files(self, src_file, logger):
        children = os.listdir(src_file)
        str_dict = {}
        for file_name in children:
            try:
                file_path = os.path.join(src_file, file_name)
                if os.path.splitext(file_path)[1] == ".txt":
                    with open(file_path, "r") as file:
                        lines = file.readlines()
                        full_str = ""
                        for line in lines:
                            if str.strip(line) == "":
                                continue
                            full_str += line
                        str_dict[self.get_time_by_filename(file_name)] = full_str
            except Exception as err:
                logger.error("read_files occurred the below error:" + file_path)
                logger.exception(err)
        return str_dict

    def get_info_dict(self, folder_path, logger):
        # folder name: cs_1_1_1_zhanghm_1010000001_0_98.254.1.151_db.ITMUSER_A_98.1.31.232_1521
        path_name = folder_path[folder_path.rfind(os.path.sep) + 1:]
        info_list = path_name.split("_")
        if len(info_list) > 10:
            each_data = {"username": info_list[4], "account": info_list[8][3:] + "_" + info_list[9],
                         "client_ip": info_list[10],
                         "db_name": info_list[7]}
            return each_data
        else:
            logger.error("input path don't have enough user account info, path:" + folder_path)
            return {}

    def start_merge(self, src_file, logger):
        logger.info("file merge begin: " + src_file)

        str_dict = self.read_all_files(src_file, logger)
        logger.info("read_all_files end, file number: " + str(len(str_dict)))

        cmd_dict = self.merge_files(str_dict, logger)
        logger.info("merge_files end, result dict number: " + str(len(cmd_dict)))

        info_dict = self.get_info_dict(src_file, logger)

        json_dict = self.parse_to_json(cmd_dict, info_dict, logger)
        logger.info("parse_to_jsone end, json command count: " + str(len(json_dict["data"])))
        return json_dict


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print("please add zip_file_name and src_file_folder, like： python xxx.py
    # cs_1_1_1_zhanghm_1010000001_0_98.254.1.151_db.ITMUSER_A_98.1.31.232_1521.rar /tmp/files/")
    #     exit()
    # src_file = sys.argv[1]
    logger = get_logger("sql_area_finder_log", "DEBUG")
    src_file = 'E://ocr//text//sql//cs_1_1_1_zhanghm_1010000001_0_98.254.1.151_db.ITMUSER_A_98.1.31.232_1521'

    conf = {'keep_items_count': 5}
    try:
        fm = FileMerger(conf)
        fm.start_merge(src_file, logger)
    except Exception as err:
        logger.error("Occurred the below error:")
        logger.exception(err)
    raise err
