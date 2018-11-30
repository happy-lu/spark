import os

os.environ["PYSPARK_PYTHON"] = "D://Python3//python.exe"

from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


#
# def show_as_line(rdd):
#     time_list = rdd.map(lambda p: (p[0], p[1])).groupByKey().take(1)
#     time_array = [y.data for (x, y) in time_list][0]
#     print(time_array)
#
#     values_rdd = rdd.map(lambda p: (p[0], (p[1], p[2]))).groupByKey()
#     r_map = values_rdd.map(
#         lambda data: parse_group_result(time_array, data[0], data[1])).collectAsMap()
#
#     for a, b in r_map.items():
#         plt.plot(time_array, b, "x-", label="status:" + str(a))
#
#     plt.xlabel('time')
#     plt.ylabel('count')
#
#     """open the grid"""
#     plt.grid(True)
#     plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
#     plt.show()


def read_data(sql_context):
    files = ["data//category_tree.csv", "data//events.csv", "data//item_properties_unique.csv"];
    category_frame = sql_context.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").options(
        header='true', inferschema='true').load(
        files[0])
    event_frame = sql_context.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").options(
        header='true', inferschema='true').load(
        files[1])
    item_frame = sql_context.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").options(
        header='true', inferschema='true').load(
        files[2])
    category_frame.registerTempTable("category_table")
    event_frame.registerTempTable("event_table")
    item_frame.registerTempTable("item_table")

    print(str(category_frame.count()) + " " + str(event_frame.count()) + " " + str(item_frame.count()) + " ")


def uniqe_items():
    read_data(sql_context)

    count_frame = sql_context.sql(
        """
        select timestamp,itemid,property,value from (
        select *,dense_rank() over (partition by itemid order by timestamp desc) as rk from item_table) t1 where rk=1
       """
    )
    print(count_frame.count())
    count_frame.show()
    count_frame.coalesce(1).write.option("header", "true").csv("data//item_properties_unique", mode='overwrite')


def init_als_data_frame(file_path):
    read_data(sql_context)

    als_data_frame = sql_context.sql(
        """
        select t1.categoryid, t1.parentid, t2.visitorid, t2.timestamp,t2.itemid,t3.property,t3.value from
        event_table t2 left join category_table t1 on t1.categoryid=t2.itemid left join item_table t3
          on t2.itemid=t3.itemid
         where t2.event='transaction'
       """
    )
    print(als_data_frame.count())
    als_data_frame.show()
    als_data_frame.coalesce(1).write.option("header", "true").csv(file_path, mode='overwrite')


def analize_event(file_path):
    read_data(sql_context)

    # als_data_frame = sql_context.sql(
    #     """
    #     select visitorid,itemid,count(*) from (
    #     select t2.visitorid, t2.timestamp,t2.itemid from
    #     event_table t2
    #      where t2.event='transaction') t group by itemid,visitorid having count(*)>1
    #    """
    # )
    # als_data_frame = sql_context.sql(
    #     """
    #     select visitorid,all_cnt,dp_cnt, itemid, timestamp from (
    #     select *, count(itemid) over (partition by visitorid) as all_cnt
    #      , count(itemid) over (partition by visitorid, itemid) as dp_cnt
    #      from event_table where event='transaction') t
    #     where all_cnt>1 order by all_cnt desc
    #    """
    # )
    als_data_frame = sql_context.sql(
        """
        select  visitorid,itemid, case when event = 'view' then 1
                                when event = 'addtocart' then 4
                                when event = 'transaction' then 10
                                else 0 end as rate,timestamp from event_table
       """
    )
    print(als_data_frame.count())
    (training, test) = als_data_frame.dropDuplicates().randomSplit([0.7, 0.3])
    als_data_frame.show()
    training.coalesce(1).write.option("header", "true").csv(file_path+"_train", mode='overwrite')
    test.coalesce(1).write.option("header", "true").csv(file_path+"_test", mode='overwrite')


def run_spark_als(file_path):
    read_data(sql_context)

    als_data_frame = sql_context.sql(
        """
        select  visitorid,itemid, case when event = 'view' then 1
                                when event = 'addtocart' then 5
                                when event = 'transaction' then 10
                                else 0 end as rate from event_table
       """
    )
    print(als_data_frame.count())
    als_data_frame.show()
    (training, test) = als_data_frame.randomSplit([0.7, 0.3])

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    base_reg = 0.01
    for iterNum in range(1):
        for regParm in range(1):
            als = ALS(maxIter=iterNum + 1, regParam=0.3, implicitPrefs=False, userCol="visitorid",
                      itemCol="itemid",
                      ratingCol="rate",
                      coldStartStrategy="drop")
            model = als.fit(training)

            # Evaluate the model by computing the RMSE on the test data
            predictions = model.transform(test)
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="rate",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print("iterNum: %s, regParam: %s, Root-mean-square error = %s" % (iterNum, base_reg, str(rmse)))
            base_reg += 0.1

    model.itemFactors.show()
    model.userFactors.show()


    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(10)
    # Generate top 10 user recommendations for each movie
    movieRecs = model.recommendForAllItems(10)

    # Generate top 10 movie recommendations for a specified set of users
    users = als_data_frame.select(als.getUserCol()).distinct().limit(3)
    userSubsetRecs = model.recommendForUserSubset(users, 10)
    # Generate top 10 user recommendations for a specified set of movies
    movies = als_data_frame.select(als.getItemCol()).distinct().limit(3)
    movieSubSetRecs = model.recommendForItemSubset(movies, 10)
    # $example off$
    userRecs.show(20, False)
    movieRecs.show(20, False)
    userSubsetRecs.show(20, False)
    movieSubSetRecs.show(20, False)


if __name__ == '__main__':
    conf = SparkConf().setAppName("my_als").setMaster("local[10]")
    conf.set("mytest.sql.crossJoin.enabled", True)
    conf.set("mytest.sql.shuffle.partitions", 5)
    conf.set("mytest.defalut.parallelism", 10)
    conf.set("mytest.driver.memory", "2g")
    spark = SparkContext(conf=conf)

    sql_context = SQLContext(spark)
    # sc.setLogLevel("DEBUG")

    file_name = "data//als_data_frame_big"
    # als_data_frame = sql_context.read.format("csv").option("header", "true").load(file_name)
    # als_data_frame.show()

    # init_als_data_frame(file_name)
    analize_event("data//event_tagged")
    # run_spark_als(sql_context)
