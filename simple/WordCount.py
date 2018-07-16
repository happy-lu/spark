import os

os.environ["PYSPARK_PYTHON"] = "D://Python3//python.exe"
# os.environ["hadoop.home.dir"]= "D:\\hadoop-3.0.3"
from pyspark import SparkContext, SparkConf

if __name__ == '__main__':
    conf = SparkConf().setAppName("test").setMaster("local[5]")
    sc = SparkContext(conf=conf)

    distFile = sc.textFile("data.txt")
    lineLengths = distFile.map(lambda s: len(s))
    print(lineLengths.reduce(lambda a, b: str(a) + " " + str(b)))

    totalLength = lineLengths.reduce(lambda a, b: a + b)
    print(totalLength)
