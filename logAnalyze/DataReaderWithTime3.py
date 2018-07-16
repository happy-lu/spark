import os

os.environ["PYSPARK_PYTHON"] = "D://Python37//python.exe"

import re
import time
# os.environ["hadoop.home.dir"]= "D:\\hadoop-3.0.3"
from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import *

import matplotlib.pyplot as plt

RESULT_FILE_SPLIT = "@@@"


def parseLine(cells):
    try:
        info = "";
        for i in range(len(cells)):
            if i > 3:
                info = info + " " + (str(cells[i]))
        return cells[0] + " " + cells[1], cells[2], int(cells[3]), str.strip(info)
    except Exception as err:
        print(cells)
        print(err)


global sql_context
global data_frame
sql_context = None
data_frame = None


def initSQLContext(sc, file_name):
    global sql_context
    global data_frame

    if not sql_context and not data_frame:
        print("begin init data_frame and sql_context")
        dist_file = sc.textFile(file_name)
        line_rdds = dist_file.filter(lambda line: re.match('\d{4}-\d{2}-\d{2}.+', line)).map(
            lambda line: re.split(" +", line)).filter(
            lambda cells: len(cells) >= 5).filter(lambda cells: cells[0].startswith('2018-07'))
        line_rows = line_rdds.map(lambda cells: parseLine(cells)).map(lambda p: Row(
            timeStr=p[0],
            threadId=p[1],
            logStatus=p[2],
            info=p[3]
        ))
        sql_context = SQLContext(sc)
        data_frame = sql_context.createDataFrame(line_rows)
        data_frame.registerTempTable("log_table")
        # data_frame.cache()
        sql_context.cacheTable("log_table")

    return sql_context, data_frame


def show_info_count(file_name, exec_sql, chart_desc, legend_desc):
    rdd_name = file_name.replace('logs', 'rdds//3d')
    rdd_name = rdd_name.replace('*', '_') + '_' + chart_desc
    if os.path.exists(rdd_name):
        rdd = sc.textFile(rdd_name).map(lambda line: re.split(RESULT_FILE_SPLIT, line))
    else:
        sql_context, data_frame = initSQLContext(sc, file_name)
        print("File: " + file_name)

        # print ( "All Count:")
        # print ( sql_context.sql("select count(*) from log_table").show())

        # print ( "Error Count:")
        # print ( sql_context.sql("select count(*) from log_table where info like '%error%'").show())
        #
        # print ( "failed Count:")
        # print ( sql_context.sql("select count(*) from log_table where info like '%failed%'").show())

        # @@@later use to_date
        d1 = time.time()
        status_cnt_frame = sql_context.sql(exec_sql)
        status_cnt_frame.cache()

        print(status_cnt_frame.take(100))
        print("method1 use time: " + str(time.time() - d1))

        rdd = status_cnt_frame.rdd
        rdd.map(lambda line: RESULT_FILE_SPLIT.join(str(i) for i in line)).repartition(
            1).saveAsTextFile(
            rdd_name)

    show_as_line(rdd, legend_desc)


def show_as_line(rdd, legend_desc):
    time_array = rdd.map(lambda p: p[1]).distinct(1).collect()
    time_array.sort()
    print(time_array)

    values_rdd = rdd.map(lambda p: (str(p[0]), (str(p[1]), str(p[2])))).groupByKey()
    r_map = values_rdd.map(
        lambda data: parse_group_result(time_array, data[0], data[1])).collectAsMap()

    fig, ax = plt.subplots()

    ax.set_xlabel('Time')
    ax.set_ylabel('Count')

    # yticks = range(-1, 10, 1)
    # ax.set_yticks(yticks)
    # ax.set_ylim([0, 10])

    for a, b in r_map.items():
        ax.plot(time_array, b, "-", label=legend_desc + str(a))

    """open the grid"""
    plt.grid(True)
    plt.legend()
    plt.show()


def parse_group_result(time_array, status, tuple_list):
    print("\nstatus:" + str(status))

    dict = {}
    [dict.setdefault(each_time, each_Value) for each_time, each_Value in tuple_list]
    print("tuple_dict:")
    print(dict)

    value_array = []
    for data_time in time_array:
        if data_time in dict:
            value = dict.get(data_time)
            value_array.append(int(value) if value != 'None' else 0)
        else:
            value_array.append(0)

    print("value_array:")
    print(value_array)
    return status, value_array

    # data_array = data_array[data_array[:, 0].argsort()]
    # t_array = map(list, zip(*data_array))[1]
    # print ( "after transfer:" + ",".join(t_array)
    # return status, t_array


if __name__ == '__main__':
    conf = SparkConf().setAppName("logReader").setMaster("local[10]")
    conf.set("spark.sql.crossJoin.enabled", True)
    conf.set("spark.sql.shuffle.partitions", 5)
    conf.set("spark.defalut.parallelism", 10)
    sc = SparkContext(conf=conf)
    # sc.setLogLevel("DEBUG")

    # files=["E://logs//ceph//ucsm-osd.37.log","E://logs//ceph//ucsm-osd.39.log",
    #        "E://logs//ceph//ucsm-osd.42.log", "E://logs//ceph//ucsm-osd.43.log"]

    # files = ["E://logs//ceph//ucsm-osd.*.log"];
    files = ["E://logs//ceph//ucsm*_ceph//ucsm-osd.*.log*"];

    exec_sql = "select logStatus, substring(timeStr,6,5) as hourStr, count(*) as cnt " \
               "from log_table group by logStatus,hourStr order by logStatus,hourStr asc"
    for each_file in files:
        show_info_count(each_file, exec_sql, "status", "status:")

    exec_sql = "select substring_index(info,'=',-1) as logStatus, substring(timeStr,6,5) as hourStr, count(*) as cnt " \
               "from log_table where info like '%existing_state=%' group by substring_index(info,'=',-1), hourStr order by logStatus,hourStr asc"
    for each_file in files:
        show_info_count(each_file, exec_sql, "EX_STATE", "EX_STATE:")

    # format is: s=STATE_ACCEPTING_WAIT_CONNECT_MSG_AUTH pgs=0 cs=0 l=0)
    exec_sql = "select SUBSTR(info,LOCATE('s=STATE_',info)+8,LOCATE('pgs=',info)-LOCATE('s=STATE_',info)-9) as logStatus, substring(timeStr,6,5) as hourStr, count(*) as cnt " \
               "from log_table where info like '%s=STATE_%' group by SUBSTR(info,LOCATE('s=STATE_',info)+8,LOCATE('pgs=',info)-LOCATE('s=STATE_',info)-9), hourStr order by logStatus,hourStr asc"
    for each_file in files:
        show_info_count(each_file, exec_sql, "STATE", "STATE:")
