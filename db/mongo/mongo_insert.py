import pymongo
from pymongo import InsertOne, DeleteOne, ReplaceOne
import time, datetime
from copy import deepcopy
import json, sys
import random
import hashlib


def insert(data_list, host, port):
    global db
    ctime = time.time()

    try:
        result = db.unis_test.bulk_write(data_list)
        print("inserted_count:", result.inserted_count)
        print("use time:", (time.time() - ctime))

    except Exception as err:
        client = pymongo.MongoClient(host=host, port=int(port),
                                     serverSelectionTimeoutMS=300000)
        db = client.jjt
        print(err)
        time.sleep(30)


def init_data(source_dict, target_batch_num):
    rlist = []
    i = 0

    while i < target_batch_num:
        ndict = deepcopy(source_dict)
        # ndict["k9"] = ndict["k9"] + "_" + str(random.randint(0, 99999999))
        random_str = str(random.randint(0, 99999999))
        ndict['filename'] = ndict['filename'].replace('.txt', "." + random_str + '.txt')
        ndict['hash'] = hashlib.sha1(random_str.encode()).hexdigest()
        ndict['size'] = str(random.randint(0, 99999))
        ndict['sequencer'] = random_str
        ndict['objectPath'] = ndict['objectPath'].replace('.txt', "." + random_str + '.txt')
        ndict['urlName'] = ndict['urlName'].replace('.txt', "." + random_str + '.txt')
        ndict['customMetadata']['objname'] = ndict['filename']
        ndict['eventTime'] = datetime.datetime.now()
        rlist.append(InsertOne(ndict))
        i += 1

    return rlist


def insert_data(index_name, input_dict, host, port, batch_num=1, loop_times=1):
    start_time = time.time()

    for i in range(loop_times):
        print("run time:", i)
        bacth_list = init_data(input_dict, batch_num)
        insert(bacth_list, host, port)

    t = time.time() - start_time
    print("index_name: %s, batch_num: %s, loop_times: %s, total use time: %s" % (
        index_name, batch_num, loop_times, t))


def read_file(file_path):
    with open(file_path, "r", encoding='UTF-8') as file:
        lines = file.read()
        data_dict = json.loads(lines)
        return data_dict


if __name__ == '__main__':
    host = sys.argv[1]
    port = sys.argv[2]
    batch_num = 100
    loop_times = 270

    client = pymongo.MongoClient(host=host, port=int(port),
                                 serverSelectionTimeoutMS=300000)
    db = client.uniswdc

    data_dict = read_file("data")
    insert_data("voice", data_dict, host, port, batch_num, loop_times)
