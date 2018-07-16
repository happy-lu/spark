import os

os.environ["PYSPARK_PYTHON"] = "D://Python3//python.exe"

import re
# os.environ["hadoop.home.dir"]= "D:\\hadoop-3.0.3"
from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import Row, SQLContext

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

    return sql_context


def pie_pct_format(value):
    """ Determine the appropriate format string for the pie chart percentage label
    Args:
        value: value of the pie slice
    Returns:
        str: formated string label; if the slice is too small to fit, returns an empty string for label
    """
    return '' if value < 7 else '%.0f%%' % value


def show_info_count(file_name):
    rdd_name = file_name.replace('logs', 'rdds//2d')
    if os.path.exists(rdd_name):
        rdd = sc.textFile(rdd_name).map(lambda line: re.split(RESULT_FILE_SPLIT, line))
    else:
        sql_context = initSQLContext(sc, file_name)
        print("File: " + file_name)

        # print ("Error Count:"
        # print (sql_context.sql("select count(*) from log_table where info like '%error%'").show()
        #
        # print ("failed Count:"
        # print (sql_context.sql("select count(*) from log_table where info like '%failed%'").show()

        print("Status Count")
        status_frame = sql_context.sql(
            "select logStatus,count(*) as cnt from log_table group by logStatus order by logStatus asc")
        print(status_frame.show())

        status_frame.rdd.map(lambda line: RESULT_FILE_SPLIT.join(str(i) for i in line)).repartition(1).saveAsTextFile(
            rdd_name)
        rdd = status_frame.rdd

    # show_as_pie(status_frame)
    show_as_line(rdd)


def show_as_pie(rdd):
    labels = rdd.map(lambda p: int(p[0])).collect();
    print(labels)
    values = rdd.map(lambda p: int(p[1])).collect();
    print(values)
    fig = plt.figure(figsize=(5, 5), facecolor='white', edgecolor='white')
    colors = ['yellowgreen', 'lightskyblue', 'gold', 'purple', 'lightcoral', 'yellow', 'black']
    # explode = (0.05, 0.05, 0.1, 0, 0, 0, 0, 0, 0, 0)
    patches, texts, autotexts = plt.pie(values, labels=labels, colors=colors,
                                        explode=None, autopct=pie_pct_format,
                                        shadow=False, startangle=125)
    for text, autotext in zip(texts, autotexts):
        if autotext.get_text() == '':
            text.set_text('')  # If the slice is small to fit, don't show a text label
    plt.legend(labels, loc=(0, 0), shadow=True)
    plt.show()


def show_as_line(rdd):
    labels = rdd.map(lambda p: int(p[0])).collect();
    print(labels)
    values = rdd.map(lambda p: int(p[1])).collect();
    print(values)
    fig = plt.figure(figsize=(8, 4.2), facecolor='white', edgecolor='white')
    plt.axis([0, max(labels), 0, max(values) + 2])
    plt.grid(b=True, which='major', axis='y')
    plt.xlabel('status')
    plt.ylabel('count')
    plt.plot(labels, values)
    plt.show()


if __name__ == '__main__':
    conf = SparkConf().setAppName("logReader").setMaster("local[50]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("INFO")

    # files=["E://logs//ceph//ucsm-osd.37.log","E://logs//ceph//ucsm-osd.39.log",
    #        "E://logs//ceph//ucsm-osd.42.log", "E://logs//ceph//ucsm-osd.43.log"]

    files = ["E://logs//ceph//ucsm-osd.37.log"];

    for each_file in files:
        show_info_count(each_file)

    # totalLength = lineLengths.reduce(lambda a, b: a + b)
    # print (totalLength)
