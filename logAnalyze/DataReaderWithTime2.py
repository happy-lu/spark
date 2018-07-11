import os

os.environ["PYSPARK_PYTHON"] = "D://Python37//python.exe"

import re
import time
# os.environ["hadoop.home.dir"]= "D:\\hadoop-3.0.3"
from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import *

import matplotlib.pyplot as plt
import numpy

RESULT_FILE_SPLIT = "@@@"


def parseLine(cells):
    info = "";
    for i in range(len(cells)):
        if i > 3:
            info = info + " " + (str(cells[i]))
    return cells[0] + " " + cells[1], cells[2], int(cells[3]), str.strip(info)


def initSQLContext(sc, file_name):
    dist_file = sc.textFile(file_name)
    line_rdds = dist_file.map(lambda line: re.split(" +", line))
    line_rows = line_rdds.map(lambda cells: parseLine(cells)).map(lambda p: Row(
        timeStr=p[0],
        threadId=p[1],
        logStatus=p[2],
        info=p[3]
    ))
    sql_context = SQLContext(sc)
    data_frame = sql_context.createDataFrame(line_rows)
    data_frame.registerTempTable("log_table")

    return sql_context, data_frame


def pie_pct_format(value):
    """ Determine the appropriate format string for the pie chart percentage label
    Args:
        value: value of the pie slice
    Returns:
        str: formated string label; if the slice is too small to fit, returns an empty string for label
    """
    return '' if value < 7 else '%.0f%%' % value


def show_info_count(file_name):
    rdd_name = file_name.replace('logs', 'rdds//3d')
    if os.path.exists(rdd_name):
        rdd = sc.textFile(rdd_name).map(lambda line: re.split(RESULT_FILE_SPLIT, line))
    else:
        sql_context, data_frame = initSQLContext(sc, file_name)
        print("File: " + file_name)

        # print ( "Error Count:")
        # print ( sql_context.sql("select count(*) from log_table where info like '%error%'").show())
        #
        # print ( "failed Count:")
        # print ( sql_context.sql("select count(*) from log_table where info like '%failed%'").show())

        # @@@later use to_date
        d1 = time.time()
        status_cnt_frame = sql_context.sql(
            "select tt1.*,tt2.cnt from (select logStatus,hourStr from (select distinct substring(timeStr,0,13) as hourStr from log_table) t1,"
            " (select distinct logStatus from log_table) t2) tt1 left join "
            "(select substring(timeStr,0,13) as hourStr, logStatus, count(*) as cnt "
            "from log_table group by logStatus,hourStr ) tt2 on tt1.logStatus=tt2.logStatus and tt1.hourStr=tt2.hourStr"
            "  order by tt2.logStatus, tt1.hourStr asc")

        print(status_cnt_frame.take(100))
        print("method1 use time: " + str(time.time() - d1))

        rdd = status_cnt_frame.rdd
        rdd.map(lambda line: RESULT_FILE_SPLIT.join(str(i) for i in line)).repartition(
            1).saveAsTextFile(
            rdd_name)

    # show_as_pie(status_frame)
    show_as_line(rdd)


def parse_group_result(time_array, status, tuple_list):
    print("\nstatus:" + str(status))

    dict = {}
    [dict.setdefault(each_time, each_Value) for each_time, each_Value in tuple_list]
    print("tuple_dict:")
    print(dict)

    value_array = []
    for data_time in time_array:
        value = dict.get(data_time)
        value_array.append(value if value != 'None' else '0')

    print("value_array:")
    print(value_array)
    return status, value_array

    # data_array = data_array[data_array[:, 0].argsort()]
    # t_array = map(list, zip(*data_array))[1]
    # print ( "after transfer:" + ",".join(t_array)
    # return status, t_array


def show_as_line(rdd):
    time_list = rdd.map(lambda p: (p[0], p[1])).groupByKey().take(1)
    time_array = [y.data for (x, y) in time_list][0]
    print(time_array)

    values_rdd = rdd.map(lambda p: (p[0], (p[1], p[2]))).groupByKey()
    r_map = values_rdd.map(
        lambda data: parse_group_result(time_array, data[0], data[1])).collectAsMap()

    for a, b in r_map.items():
        plt.plot(time_array, b, "x-", label="status:" + str(a))

    plt.xlabel('time')
    plt.ylabel('count')

    """open the grid"""
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    conf = SparkConf().setAppName("logReader").setMaster("local[10]")
    conf.set("spark.sql.crossJoin.enabled", True)
    conf.set("spark.sql.shuffle.partitions", 5)
    conf.set("spark.defalut.parallelism", 2)
    sc = SparkContext(conf=conf)
    # sc.setLogLevel("DEBUG")

    # files=["E://logs//ceph//ucsm-osd.37.log","E://logs//ceph//ucsm-osd.39.log",
    #        "E://logs//ceph//ucsm-osd.42.log", "E://logs//ceph//ucsm-osd.43.log"]

    files = ["E://logs//ceph//ucsm-osd.100.log"];

    for each_file in files:
        show_info_count(each_file)

    # totalLength = lineLengths.reduce(lambda a, b: a + b)
    # print ( totalLength)
