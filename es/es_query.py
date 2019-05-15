import time
from elasticsearch import Elasticsearch
import random
import threading


def query_data(es, index_name, loop_times=1):
    query_str = """
{
  "query": {
    "bool": {
      "must": [
        {
          "wildcard": {
            "name": {
              "value": "*@@@@@@"
            }
          }
        }
      ]
    }
  },
  "post_filter": {
    "range": {
      "last_modified": {
        "gte": 1540509685
      }
    }
  }
}
"""
    took_sum = 0
    for i in range(loop_times):
        index = str(random.randint(0, 999999))
        print("need query index:"+ index)
        new_str = query_str.replace("@@@@@@", index)
        data = eval(new_str)

        try:
            res = es.search(
                index=index_name,
                timeout='15s',
                body=data
            )
            took_sum += res['took']
            print("index: %s, took: %s, count: %s" % (index, res['took'], res['hits']['total']))
            time.sleep(5)
        except Exception as e:
            print("aaa")
            print(e)

    print(str(time.time()) + ", average took: " + str(took_sum / loop_times))


if __name__ == '__main__':
    # es = Elasticsearch(['192.168.232.210:9200','192.168.232.213:9200', '192.168.232.212:9200'])
    es = Elasticsearch(['192.168.221.23:9200', '192.168.221.25:9200'])
    for i in range(1):
        t = threading.Thread(target=query_data, args=(es, "som_perf_small", 100))
        t.start()
    print(str(time.time()) + ",begin")

    # es = Elasticsearch(['192.168.221.25:9200'])
    # for i in range(25):
    #     t = threading.Thread(target=query_data, args=(es, "som_perf_small", 10))
    #     t.start()
