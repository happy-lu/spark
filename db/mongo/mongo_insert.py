import pymongo
from pymongo import InsertOne, DeleteOne, ReplaceOne
import time
from copy import deepcopy
import random

client = pymongo.MongoClient(host='192.168.232.130,192.168.232.131,192.168.232.132', port=27017, serverSelectionTimeoutMS=300000)
db = client.jjt


def insert(data_list):
    global db
    ctime = time.time()

    try:
        result = db.test2.bulk_write(data_list)
        print("inserted_count:", result.inserted_count)
        print("use time:", (time.time() - ctime))
    except Exception as err:
        client = pymongo.MongoClient(host='192.168.232.130,192.168.232.131,192.168.232.132', port=27017, serverSelectionTimeoutMS=300000)
        db = client.jjt
        print(err)
        time.sleep(30)




def init_data(source_dict, target_batch_num):
    rlist = []
    i = 0

    while i < target_batch_num:
        ndict = deepcopy(source_dict)
        ndict["k9"] = ndict["k9"] + "_" + str(random.randint(0, 99999999))
        rlist.append(InsertOne(ndict))
        i += 1

    return rlist


def insert_data(index_name, input_dict, batch_num=1, loop_times=1):
    start_time = time.time()

    for i in range(loop_times):
        print("run time:", i)
        bacth_list = init_data(input_dict, batch_num)
        insert(bacth_list)

    t = time.time() - start_time
    print("index_name: %s, batch_num: %s, loop_times: %s, total use time: %s" % (
        index_name, batch_num, loop_times, t))


def read_file(file_path):
    with open(file_path, "r", encoding='UTF-8') as file:
        lines = file.read()
        data_dict = eval(lines)
        return data_dict


if __name__ == '__main__':
    batch_num = 10000
    loop_times =5000

    data_dict = read_file("data")
    insert_data("voice", data_dict, batch_num, loop_times)
