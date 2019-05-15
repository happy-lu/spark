from mongo_oper import *
from logging_util import *
from kafka_msg_oper import *
from config_reader import *


class DataSyncOper():
    def __init__(self, conf_file_str):
        self.config = ConfigReader(conf_file_str)
        self.logger = get_logger("data_sync_log", self.config.get_conf_str("log_level").upper())

    def deal_with_kafka_msg(self):
        kafka_server = self.config.get_conf_str("kafka_server")
        kafka_topic = self.config.get_conf_str("kafka_topic")
        kafka_group = self.config.get_conf_str("kafka_group")

        mongo_server = self.config.get_conf_str("mongo_server")
        mongo_db_name = self.config.get_conf_str("mongo_db_name")
        mongo_db_port = self.config.get_conf_str("mongo_db_port")

        mongo = MongoOper(mongo_server, mongo_db_name, self.logger, int(mongo_db_port))

        skc = BatchKafkaConsumer(kafka_server, kafka_topic, kafka_group, self.logger)
        datas = skc.consume()
        for message in datas:
            # logger.info(
            #     "%s, %d-%d, key=%s value=%s" % (
            #         message.topic, message.partition, message.offset, message.key, message.value))
            try:
                data_list = mongo.get_insert_data(message)
                mongo.batch_insert(data_list)
                self.logger.debug("Success deal with %s message" % (len(data_list)))
            except Exception as error:
                self.logger("deal_with_kafka_msg get error")
                self.logger.exception(error)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("please input the config file name, like： python xxx.py xxx.conf")
        exit()

    dso = DataSyncOper(sys.argv[1])
    dso.deal_with_kafka_msg()
