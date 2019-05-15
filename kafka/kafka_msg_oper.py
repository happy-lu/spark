from kafka import KafkaConsumer


class SimpleKafkaConsumer():
    # if some process need long time to deal with consume message, we need set auto_commit=False
    # and make the timeout_ms> process_time
    def __init__(self, server_str, topic, group_name, logger, auto_commit=True, timeout_ms=60000):
        self.kafka_servers = server_str.split(",")
        self.logger = logger
        self.topic = topic
        self.group_name = group_name
        self.auto_commit = auto_commit
        self.consumer = KafkaConsumer(topic, bootstrap_servers=self.kafka_servers, group_id=group_name,
                                      enable_auto_commit=auto_commit, fetch_max_wait_ms=1000,
                                      request_timeout_ms=timeout_ms + 30000,
                                      session_timeout_ms=timeout_ms,
                                      heartbeat_interval_ms=2000)

    def consume(self):
        try:
            for message in self.consumer:
                yield message
                if not self.auto_commit:
                    self.consumer.commit_async()
                    self.logger.info("commit the message")
        except Exception as e:
            self.logger.error("consume occurred error: topic:%s, group_name:%s" % (self.topic, self.group_name))
            self.logger.exception(e)


class BatchKafkaConsumer(SimpleKafkaConsumer):
    def __init__(self, server_str, topic, group_name, logger, batch_commit_num=100):
        self.kafka_servers = server_str.split(",")
        self.logger = logger
        self.topic = topic
        self.group_name = group_name
        self.batch_commit_num = batch_commit_num
        self.consumer = KafkaConsumer(topic, bootstrap_servers=self.kafka_servers, group_id=group_name,
                                      enable_auto_commit=False, fetch_max_wait_ms=1000, request_timeout_ms=300000,
                                      session_timeout_ms=6000,
                                      heartbeat_interval_ms=2000)

    def consume(self):
        try:
            result = []
            for message in self.consumer:
                result.append(message)
                if len(result) >= self.batch_commit_num:
                    self.consumer.commit_async()
                    # self.logger.debug("Success commit the message")
                    yield result
                    result = []
        except Exception as e:
            self.logger.error("consume occurred error: topic:%s, group_name:%s" % (self.topic, self.group_name))
            self.logger.error(e)
