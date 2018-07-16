import os

os.environ["PYSPARK_PYTHON"] = "D://Python3//python.exe"

import re
import time
# os.environ["hadoop.home.dir"]= "D:\\hadoop-3.0.3"
from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import *

import matplotlib.pyplot as plt

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

        # print ("Error Count:")
        # print (sql_context.sql("select count(*) from log_table where info like '%error%'").show())
        #
        # print ("failed Count:")
        # print (sql_context.sql("select count(*) from log_table where info like '%failed%'").show())

        # Method 1
        d1 = time.time()
        all_frame = sql_context.sql(
            "select logStatus,hourStr from (select distinct substring(timeStr,0,13) as hourStr from log_table) t1,"
            " (select distinct logStatus from log_table) t2")
        print(all_frame.take(100))
        print("method1 use time: " + str(time.time() - d1))

        # # Method 2
        # print ("Status Count")
        # d1 = time.time()
        # time_frame = data_frame.selectExpr("substring(timeStr,0,13) as hourStr").distinct()
        # print (time_frame.take(100))
        #
        # status_frame = data_frame.select("logStatus").distinct()
        # print (status_frame.take(100))
        #
        # # map(lambda p: [p[0] + "," + str(p[1]), 0])
        # all_frame = time_frame.crossJoin(status_frame)
        # # .select("hourStr", "logStatus",(status_frame.logStatus * 0).alias("dCnt"))
        # print (all_frame.take(100))
        # print ("method2 use time: " + str(time.time() - d1))

        # @@@later use to_date
        print(sql_context.sql("select count(*) from log_table").show())
        status_cnt_frame = sql_context.sql(
            "select substring(timeStr,0,13) as hourStr, logStatus, count(*) as cnt "
            "from log_table group by logStatus,hourStr order by logStatus,hourStr asc")

        # status_cnt_frame = status_cnt_frame.map(lambda p: [p[0] + "," + str(p[1]), p[2]])
        final_frame = all_frame.join(status_cnt_frame, ["logStatus", "hourStr"], "left")
        print(final_frame.take(100))

        rdd = final_frame.rdd
        rdd.map(lambda line: RESULT_FILE_SPLIT.join(str(i) for i in line)).repartition(
            1).saveAsTextFile(
            rdd_name)

    # show_as_pie(status_frame)
    show_as_line(rdd)


def parse_group_result(a, b):
    print("a:" + str(a))
    data_array = [str(each_b if each_b != 'None' else '0') for each_b in b]
    print("b:")
    print(data_array)

    return a, data_array


def show_as_line(rdd):
    time_list = rdd.map(lambda p: (p[0], p[1])).groupByKey().take(1)
    time_array = [y.data for (x, y) in time_list][0]
    print(time_array)

    values_rdd = rdd.map(lambda p: (p[0], p[2])).groupByKey()
    # print (values_rdd.take(1))
    # result= [y for (x, y) in values_rdd]
    # print (result)

    r_map = values_rdd.map(
        lambda a_b: parse_group_result(a_b[0], a_b[1])).collectAsMap()

    for a, b in r_map.items():
        plt.plot(time_array, b, "x-", label="status:" + str(a))

    plt.xlabel('time')
    plt.ylabel('count')

    # plt.ylim(0,10)
    # yticks = range(0, 100, 10)
    # plt.yticks(yticks)

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
    # print (totalLength)
