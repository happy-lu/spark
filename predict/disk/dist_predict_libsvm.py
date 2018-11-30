import os
import time
import re
# $example on$
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# $example off$
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark import *
from pyspark.sql import *
from pyspark.mllib.util import *


def exec_method(spark, data):
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = \
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexer.labels)

    # dtree
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])
    model = pipeline.fit(trainingData)

    # gbt
    # gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)
    # pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt, labelConverter])
    # model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    print("predictedLabel=1: " + str(predictions.filter("predictedLabel=1.0").count()))
    print("label=1: " + str(predictions.filter("label=1.0").count()))
    print("total: " + str(predictions.count()))

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[2]
    print(rfModel)  # summary only
    # $example off$

    spark.stop()


def read_csv(spark, file_name):
    sql_context = SQLContext(spark)

    df = sql_context.read.format('com.databricks.mytest.csv').options(header='true', format="string").load(
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

    assembler = VectorAssembler(
        inputCols=f_cols,
        outputCol="indexedFeatures")
    df4 = assembler.transform(df3)

    return df4


def sl_by_libsvm(spark):
    out_folder = 'E:\\mldata\\libsvm-small'
    rdd_name = os.path.join(out_folder, file_name)

    if os.path.exists(rdd_name):
        data = spark.read.format("libsvm").load(rdd_name)
    else:
        data = read_csv(spark, os.path.join(in_folder, file_name))
        print("read one file use time:" + str(time.time() - exec_start_time))
        data = data.select("failure", "indexedFeatures")
        data = data.withColumn("failure", data["failure"].cast("double"))
        # ut = MLUtils.convertVectorColumnsToML(df, "indexedFeatures")
        # ut = ut.withColumnRenamed("failure", "label").withColumnRenamed("indexedFeatures", "features")
        # ut = ut.withColumn("label", ut["label"].cast("double"))
        data.show()
        data.write.format("libsvm").save(rdd_name)

    return data


def svm(data):
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    # svm
    lsvc = LinearSVC(maxIter=10, regParam=0.1)
    model = lsvc.fit(trainingData)

    # Print the coefficients and intercept for linear SVC
    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))

    predictions = model.transform(testData)
    predictions.select("prediction", "label", "features").show(5)

    print("prediction=1.0:" + str(predictions.filter("prediction=1.0").count()))
    print("label=1.0:" + str(predictions.filter("label=1.0").count()))
    print("total:" + str(predictions.count()))

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))


if __name__ == "__main__":

    exec_start_time = time.time()

    spark = SparkSession.builder.appName("disk_predict").config("mytest.driver.memory",
                                                                "2g").config("mytest.sql.shuffle.partitions",
                                                                             5).config("mytest.defalut.parallelism",
                                                                                       10).getOrCreate()

    in_folder = 'E:\\mldata\\hard-disk-2016-q1-data-small'
    prepare = False

    children = os.listdir(in_folder)

    positive_data = spark.read.format("libsvm").load("E:\\mldata\\libsvm-all-1")
    full_data = positive_data

    for i in range(300):
        full_data = full_data.union(positive_data)

    for file_name in children:
        data = sl_by_libsvm(spark)

        if not prepare:
            full_data = full_data.union(data)

    print("begin exec_method")

    if not prepare:
        exec_method(spark, full_data)
        # svm(full_data)

    print("use time:" + str(time.time() - exec_start_time))
