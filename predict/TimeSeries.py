import os

os.environ["PYSPARK_PYTHON"] = "D://Python3//python.exe"

import re
from pyspark import *
from pyspark.sql import *
from sparkts import *
from sparkts.datetimeindex import *
from sparkts.timeseriesrdd import time_series_rdd_from_observations

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
    data_rdd = df.select("createtime", "hostaddr", "cpu_usage")
    print(data_rdd.take(5))

    freq = DayFrequency(1, 1, sc)
    dtIndex = uniform(start='2018-06-18T00:00-00:00', end='2018-06-28T00:00-07:00', freq=freq, sc=sc)

    # Align the ticker data on the DateTimeIndex to create a TimeSeriesRDD
    cpu_rdd = time_series_rdd_from_observations(dtIndex, data_rdd, "timestamp", "hostaddr", "cpu_usage")

   # Cache it in memory
    cpu_rdd.cache()

    # Count the number of series (number of symbols)
    print(cpu_rdd.count())

    # Impute missing values using linear interpolation
    filled = cpu_rdd.fill("linear")

    # Compute return rates
    returnRates = filled.return_rates()
    print(returnRates)


    # Durbin-Watson test for serial correlation, ported from TimeSeriesStatisticalTests.scala
    def dwtest(residuals):
        residsSum = residuals[0] * residuals[0]
        diffsSum = 0.0
        i = 1
        while i < len(residuals):
            residsSum += residuals[i] * residuals[i]
            diff = residuals[i] - residuals[i - 1]
            diffsSum += diff * diff
            i += 1
        return diffsSum / residsSum


    # Compute Durbin-Watson stats for each series
    # Swap ticker symbol and stats so min and max compare the statistic value, not the
    # ticker names.
    dwStats = returnRates.map_series(lambda row: (row[0], [dwtest(row[1])])).map(lambda x: (x[1], x[0]))

    print(dwStats.min())
    print(dwStats.max())