import os

os.environ["PYSPARK_PYTHON"] = "D://Python3//python.exe"
# os.environ["hadoop.home.dir"]= "D:\\hadoop-3.0.3"
from pyspark import SparkContext, SparkConf

if __name__ == '__main__':
    conf = SparkConf().setAppName("test").setMaster("local[5]")
    sc = SparkContext(conf=conf)

    row_rdd = sc.textFile("data.txt")
    words_rdd = row_rdd.flatMap(lambda s: s.split(' '))
    print("words_rdd:" + str(words_rdd.take(100)))

    num_rdd = words_rdd.map(lambda s: (s, 1)).reduceByKey(lambda a, b: a + b)
    print("num_rdd:" + str(num_rdd.take(100)))
