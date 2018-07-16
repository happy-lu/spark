import os

os.environ["PYSPARK_PYTHON"] = "D://Python37//python.exe"

import re
# os.environ["hadoop.home.dir"]= "D:\\hadoop-3.0.3"
from pyspark import *
from pyspark.sql import Row, SQLContext

if __name__ == '__main__':
    conf = SparkConf().setAppName("aaa").setMaster("local[5]")
    conf.set("spark.sql.crossJoin.enabled", True)
    conf.set("spark.sql.shuffle.partitions", 5)
    conf.set("spark.defalut.parallelism", 2)
    sc = SparkContext(conf=conf)
    # sc.setLogLevel("DEBUG")

    sc.parallelize([2, 3, 4, 5, 6]).take(10)
    # sc.parallelize(range(100), 1).filter(lambda x: x > 90).take(5)

    line = [1, 2]
    line2 = [1, 3]

    print([str(x) + str(y) for x, y in zip(line, line2)])

    result1 = sc.parallelize(line)
    print(result1.collect())
    # print(result1.take(5))

    # result = sc.parallelize(line).cartesian(sc.parallelize(line2)).flatMap(lambda data: (data[0], data[1]))
    # result.cache()
    # print(result.collect())
    # print(result.take(100))

    # print ("@@@".join(str(i) for i in line))

    rdd = sc.parallelize([1, 1, 2, 3, 5, 8])
    result = rdd.groupBy(lambda d: d % 2).collect()
    print([(x, sorted(y)) for (x, y) in result])

    rdd = sc.parallelize(range(1, 10))
    result = rdd.reduce(lambda a, b: a + b)
    print(result)
