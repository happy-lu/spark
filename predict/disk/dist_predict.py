import os
import time
# $example on$
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# $example off$
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark import *


def exec_method(spark, data):
    labelIndexer = StringIndexer(inputCol="failure", outputCol="failure_index").fit(data)

    cols = data.columns[5:]
    assembler = VectorAssembler(
        inputCols=cols,
        outputCol="indexedFeatures")
    featureIndexer = assembler.transform(data)
    featureIndexer.select("indexedFeatures").show(truncate=False)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="failure_index", featuresCol="indexedFeatures", numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, assembler, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("predictedLabel", "failure", "indexedFeatures").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="failure_index", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[2]
    print(rfModel)  # summary only
    print("model detail\n:" + rfModel.toDebugString)
    # $example off$

    spark.stop()


def read_csv(spark, file_name):
    sql_context = SQLContext(spark)

    df = sql_context.read.format('com.databricks.spark.csv').options(header='true', format="string").load(
        file_name)

    dateIndexer = StringIndexer(inputCol="date", outputCol="date_index").fit(df)
    serialIndexer = StringIndexer(inputCol="serial_number", outputCol="serial_number_index").fit(df)
    modelIndexer = StringIndexer(inputCol="model", outputCol="model_index").fit(df)

    df1 = dateIndexer.transform(df)
    df2 = serialIndexer.transform(df1)
    df3 = modelIndexer.transform(df2)

    df3 = df3.na.fill("0")
    f_cols = df3.columns[5:]
    for name in f_cols:
        # df3 = df3.withColumn(name, "0" if df3[name] == "null" else df3[name])
        df3 = df3.withColumn(name, df3[name].cast("double"))

    # df3.show()

    return df3


if __name__ == "__main__":

    exec_start_time = time.time()
    conf = SparkConf().setAppName("disk_predict").setMaster("local[10]")
    conf.set("mytest.sql.crossJoin.enabled", True)
    conf.set("mytest.sql.shuffle.partitions", 5)
    conf.set("mytest.defalut.parallelism", 10)
    conf.set("mytest.driver.memory", "2g")
    spark = SparkContext(conf=conf)


    in_folder = 'E:\\mldata\\hard-disk-2016-q1-data-small2'
    out_folder = 'E:\\mldata\\libsvm'

    children = os.listdir(in_folder)

    full_data = None
    for file_name in children:
        data = read_csv(spark, os.path.join(in_folder, file_name))
        print("read one file use time:" + str(time.time() - exec_start_time))
        if full_data:
            full_data = full_data.union(data)
        else:
            full_data = data

    print("begin exec_method" )
    exec_method(spark, full_data)

    print("use time:" + str(time.time() - exec_start_time))
