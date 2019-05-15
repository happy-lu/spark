import time
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from copy import deepcopy
import random


def insert_data(index_name, input_list, batch_num=1, loop_times=1):
    start_time = time.time()
    es = Elasticsearch(['192.168.242.215:9200','192.168.242.216:9200','192.168.242.217:9200'])

    if batch_num == 1:
        for each_dict in input_list:
            es.index(index=index_name, doc_type='doc', body=each_dict, id=None)
    else:
        total_times = 0
        for i in range(loop_times):
            bacth_list = init_data(input_list, batch_num)

            actions = []
            for each_dict in bacth_list:
                action = {
                    "_index": index_name,
                    "_type": 'doc',
                    "_source": each_dict
                }
                actions.append(action)

            last_time = time.time()
            helpers.bulk(es, actions, request_timeout=60)
            d_time = (time.time() - last_time)

            total_times += d_time
            print("batch_num: %s, loop_times: %s, use time: %s" % (
                batch_num, i, d_time))

    t = time.time() - start_time
    print("index_name: %s, batch_num: %s, loop_times: %s, use time: %s, insert_es_time: %s" % (
        index_name, batch_num, loop_times, t, total_times))


def init_data(source_list, target_batch_num):
    rlist = []
    i = 0

    while i < target_batch_num:
        for each_dict in source_list:
            i += 1
            ndict = deepcopy(each_dict)
            ndict["last_modified"] = int(time.time())
            ndict["name"] = ndict["name"] + "_" + str(random.randint(0, 99999999))
            rlist.append(ndict)

            if i == target_batch_num:
                break
    return rlist


def read_file(file_path):
    with open(file_path, "r", encoding='UTF-8') as file:
        lines = file.read()
        data_dict = eval(lines)
        return data_dict


if __name__ == '__main__':
    batch_num = 10000
    loop_times = 10

    data_list = read_file("som.data")
    # print(data_list)

    insert_data("som_test", data_list, batch_num, loop_times)

    # voice_list = read_file("data")
    # print(voice_list[0])
