import os

os.environ["PYSPARK_PYTHON"] = "D://Python3//python.exe"

from pyspark import *
from pyspark.sql import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    conf = SparkConf().setAppName("logReader").setMaster("local[10]")
    conf.set("mytest.sql.crossJoin.enabled", True)
    conf.set("mytest.sql.shuffle.partitions", 5)
    conf.set("mytest.defalut.parallelism", 10)
    sc = SparkContext(conf=conf)
    sql_context = SQLContext(sc)
    # sc.setLogLevel("DEBUG")

    # files=["E://logs//ceph//ucsm-osd.37.log","E://logs//ceph//ucsm-osd.39.log",
    #        "E://logs//ceph//ucsm-osd.42.log", "E://logs//ceph//ucsm-osd.43.log"]

    # files = ["E://logs//ceph//ucsm-osd.*.log"];
    file = "E://mldata//predict//hostresource_nonet.csv";

    df = sql_context.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
        file)
    pandas_frame = df.filter("hostaddr='192.168.232.183'").select("createtime", "cpu_usage").toPandas().set_index(
        "createtime")

    print(pandas_frame.dtypes)
    plt.rcParams['axes.unicode_minus'] = False
    pandas_frame.plot()
    plt.show()
