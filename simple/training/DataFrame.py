import os

os.environ["PYSPARK_PYTHON"] = "D://Python3//python.exe"

import re
from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import *


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


if __name__ == '__main__':
    conf = SparkConf().setAppName("logReader").setMaster("local[10]")
    conf.set("mytest.sql.crossJoin.enabled", True)
    conf.set("mytest.sql.shuffle.partitions", 5)
    conf.set("mytest.defalut.parallelism", 10)
    sc = SparkContext(conf=conf)
    # sc.setLogLevel("DEBUG")

    file_name = "E://logs//ceph//ucsm*_ceph//ucsm-osd.*.log*";

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

    exec_sql = "select logStatus, substring(timeStr,6,5) as hourStr, count(*) as cnt " \
               "from log_table group by logStatus,hourStr order by logStatus,hourStr asc"
    status_cnt_frame = sql_context.sql(exec_sql)
    status_cnt_frame.cache()

    print(status_cnt_frame.take(100))
