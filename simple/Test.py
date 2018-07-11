import os

os.environ["PYSPARK_PYTHON"] = "D://Python37//python.exe"

import re
# os.environ["hadoop.home.dir"]= "D:\\hadoop-3.0.3"
from pyspark import *
from pyspark.sql import Row, SQLContext

if __name__ == '__main__':


    conf = SparkConf().setAppName("logReader").setMaster("local[5]")
    conf.set("spark.sql.crossJoin.enabled", True)
    conf.set("spark.sql.shuffle.partitions", 5)
    conf.set("spark.defalut.parallelism", 2)
    sc = SparkContext(conf=conf)
    # sc.setLogLevel("DEBUG")

    line = [1, 2]
    line2 = [1,3]

    print [str(x)+ str(y) for x, y in zip(line,line2)]


    result = sc.parallelize(line).cartesian(sc.parallelize(line2))
    print (result.take(100))
    RDD.persist(result)


    # print "@@@".join(str(i) for i in line)

    rdd = sc.parallelize([1, 1, 2, 3, 5, 8])
    result = rdd.groupBy(lambda d: d % 2).collect()
    print ([(x, sorted(y)) for (x, y) in result])

    rdd = sc.parallelize(range(1, 10))
    result = rdd.reduce(lambda a, b: a + b)
    print (result)


