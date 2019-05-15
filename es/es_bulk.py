import time
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from copy import deepcopy
import random


def insert_data():
    start_time = time.time()
    es = Elasticsearch(['192.168.221.23:9200'])

    # actions = []
    # each_dict = {"aaa": 1, "bbb": 2}
    # action = {
    #     "_index": "tn_custom1_tp_unis_v_1.0",
    #     "_id": "123",
    #     "_type": 'doc',
    #     "_source": each_dict
    # }
    # actions.append(action)
    #
    # last_time = time.time()
    # helpers.bulk(es, actions, request_timeout=60)

    actions = []
    each_dict = {"222": 4}
    update_dict = {"doc": each_dict, "upsert": each_dict}
    action = {
        "_index": "tn_custom1_tp_unis_v_1.0",
        "_id": "333",
        "_type": 'doc',
        "_op_type": 'update',
        "_source": update_dict
    }

    # action = [{'update':{}}, action]
    actions.append(action)

    # es.update(index="tn_custom1_tp_unis_v_1.0", doc_type='doc', id="123", body=each_dict)
    last_time = time.time()
    helpers.bulk(es, actions, request_timeout=60)


if __name__ == '__main__':
    batch_num = 1
    loop_times = 1

    insert_data()

    # voice_list = read_file("data")
    # print(voice_list[0])
