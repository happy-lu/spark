from datetime import datetime
from copy import deepcopy

import re


def parse_all_lines(lines, patterns, client_ip, user_name, account, db_name, exec_time, file_path):
    #  return data format
    # {"index":"video_analyze", "data":[{"execute_host": "host-1", "number": 0, "execute_time": 1514822400.0,
    # "command": "lscpu", "command_result": "bbb", "path": "s3://192.168.231.217:7480/host-1-0-pic.png"}]
    result = {}
    result["index"] = "video_analyze"
    result["data"] = []
    number = 0

    each_data = {"username": user_name, "account": account, "client_ip": client_ip,
                 "db_name": db_name, "execute_time": exec_time, "path": file_path,
                 "command_result": ""}
    c_data = None

    for line in lines:
        if line and line.strip() != "":
            line = line.strip()
            cur_line_is_cmd = False
            for pt in patterns:
                m_obj = re.match(pt, line, re.I)
                if m_obj:
                    cmd_str = line[m_obj.end():].strip()
                    if cmd_str and cmd_str != "":
                        # find valid command, save it
                        print(cmd_str + "\n", end='')
                        c_data = deepcopy(each_data)
                        c_data["command"] = cmd_str
                        c_data["number"] = number
                        result["data"].append(c_data)

                        number += 1
                        cur_line_is_cmd = True
                    else:
                        # empty command, skip
                        cur_line_is_cmd = True
                    break;
            if c_data and not cur_line_is_cmd:
                c_data["command_result"] = c_data["command_result"] + line + "\n"

    return result


if __name__ == '__main__':
    file_name = "clienta_testuser_zhanghm_db123_2018-08-08.txt"
    file_path = "E://ocr//text//" + file_name
    name_info = file_name.split(".")[0].split("_")
    start_date = datetime.strptime(name_info[4], "%Y-%m-%d").timestamp()

    with open(file_path, "r") as file:
        lines = file.readlines()
        result = parse_all_lines(lines, ["\[.+~\]#", "\[.+~\]\$"], name_info[0], name_info[1], name_info[2],
                                 name_info[3], start_date, file_path)
        print(result)
