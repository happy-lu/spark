from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import Row, SQLContext
import re
from operator import add

if __name__ == '__main__':
    conf = SparkConf().setAppName("logReader").setMaster("local[1]")
    conf.set("mytest.sql.crossJoin.enabled", True)
    conf.set("mytest.sql.shuffle.partitions", 5)
    conf.set("mytest.defalut.parallelism", 2)

    sc = SparkContext(conf=conf)
    sc.setLogLevel("INFO")

    input = ["A", "C"]

    str = "A:B,C,D,E,F;B:A,C,D,E;C:A,B,E;D:A,B,E;"
    #
    # rdd = sc.parallelize(re.split(";", str)[0:-1]).map(
    #     lambda kv: (re.split(":", kv)[0], re.split(":", kv)[1])). \
    #     filter(lambda kv: kv[0] in input).map(lambda kv: kv[1])

    keys = sc.parallelize(re.split(";", str)[0:-1]).map(
        lambda kv: (re.split(":", kv)[0], re.split(":", kv)[1].split(","))). \
        filter(lambda kv: kv[0] in input).flatMapValues(lambda x: x).map(lambda x: (x[1], 1)).reduceByKey(add).filter(
        lambda x: x[1] > 1).keys().collect()

    print(",".join(keys))

    # rdd1=rdd.map(lambda v: re.split(",", v)).collect()
    # print(rdd1)
