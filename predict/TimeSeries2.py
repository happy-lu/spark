import os

os.environ["PYSPARK_PYTHON"] = "D://Python3//python.exe"

from pyspark import *
from pyspark.sql import *
import matplotlib.pyplot as plt


def parse_group_result(time_array, status, tuple_list):
    print("\nstatus:" + str(status))

    print("time_array:")
    print(time_array)
    dict = {}
    [dict.setdefault(each_time, each_Value) for each_time, each_Value in tuple_list]
    print("tuple_dict:")
    print(dict)

    value_array = []
    for data_time in time_array:
        if data_time in dict:
            value = dict.get(data_time)
            value_array.append(value if value != 'None' else 0)
        else:
            value_array.append(0)

    print("value_array:")
    print(value_array)
    return status, value_array


def show_as_line(rdd, legend_desc):
    time_array = rdd.map(lambda p: p[1]).distinct(1).collect()
    time_array.sort()
    print(time_array)

    values_rdd = rdd.map(lambda p: (str(p[0]), (p[1], p[2]))).groupByKey()
    r_map = values_rdd.map(
        lambda data: parse_group_result(time_array, data[0], data[1])).collectAsMap()

    fig, ax = plt.subplots()

    ax.set_xlabel('Time')
    ax.set_ylabel('Count')

    yticks = range(0, 100, 5)
    ax.set_yticks(yticks)
    # ax.set_ylim([0, 10])

    # xticks = range(0, 100, 5)
    # ax.set_yticks(yticks)

    for a, b in r_map.items():
        ax.plot(time_array, b, "-", label=legend_desc + str(a))

    """open the grid"""
    plt.grid(True)
    plt.legend()
    plt.show()


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
    data_rdd = df.select("hostaddr", "createtime", "cpu_usage").filter("hostaddr='192.168.232.183'").rdd
    data_rdd.cache()
    print(data_rdd.take(3))
    show_as_line(data_rdd, "cpu")
