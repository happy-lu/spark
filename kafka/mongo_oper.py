import time

import pymongo
from pymongo import InsertOne


class MongoOper():

    def __init__(self, server_str, db_name, logger, server_port=27017, timeout=300000):
        self.server_str = server_str
        self.db_name = db_name
        self.server_port = server_port
        self.timeout = timeout
        self.logger = logger
        self.init_db()

    def init_db(self):
        client = pymongo.MongoClient(host=self.server_str, port=self.server_port,
                                     serverSelectionTimeoutMS=self.timeout)
        self.db = client[self.db_name]

    def batch_insert(self, data_list):
        try:
            self.db.test2.bulk_write(data_list)
        except Exception as err:
            self.init_db()
            self.logger.error("insert data occurred error: server_str:%s" % (self.server_str))
            self.logger.exception(err)
            time.sleep(30)

    def get_insert_data(self, data_arr):
        rlist = []

        for each_msg in data_arr:
            ndict = {}
            msg_attrs = each_msg.value.decode().split(',')
            for i, data_str in enumerate(msg_attrs):
                if i == 4 or i == 5:
                    data_str = time.mktime(time.strptime(data_str, "%Y-%m-%dT%H:%M:%S"))
                ndict["k" + str(i)] = data_str

            rlist.append(InsertOne(ndict))
        return rlist

