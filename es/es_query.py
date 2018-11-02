import time
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from copy import deepcopy
import random


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
        new_str = query_str.replace("@@@@@@", "741378")
        data = eval(new_str)

        res = es.search(
            index=index_name,
            request_timeout = 60,
            body=data
        )
        took_sum += res['took']
        print("index: %s, took: %s, count: %s" % (index, res['took'], res['hits']['total']))

    print("average took: " + str(took_sum / loop_times))


if __name__ == '__main__':
    es = Elasticsearch(['192.168.231.250:9200'])

    query_data(es, "r_som", 100)
