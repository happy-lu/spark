import os
import time
import re
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

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

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

    assembler = VectorAssembler(
        inputCols=f_cols,
        outputCol="indexedFeatures")
    df4 = assembler.transform(df3)

    return df4


def sl_by_libsvm(spark):
    data = read_csv(spark, os.path.join(in_folder, file_name))
    print("read one file use time:" + str(time.time() - exec_start_time))
    data = data.filter("failure=1.0")
    data = data.select("failure", "indexedFeatures")
    data = data.withColumn("failure", data["failure"].cast("double"))
    # ut = MLUtils.convertVectorColumnsToML(df, "indexedFeatures")
    # ut = ut.withColumnRenamed("failure", "label").withColumnRenamed("indexedFeatures", "features")
    # ut = ut.withColumn("label", ut["label"].cast("double"))
    data.show()

    return data


if __name__ == "__main__":

    exec_start_time = time.time()

    spark = SparkSession.builder.appName("disk_predict").config("spark.driver.memory",
                                                                "2g").getOrCreate()

    in_folder = 'E:\\mldata\\hard-disk-2016-q1-data-small'
    rdd_name = 'E:\\mldata\\libsvm-all-1'

    children = os.listdir(in_folder)

    full_data = None
    for file_name in children:
        data = sl_by_libsvm(spark)

        if full_data:
            full_data = full_data.union(data)
        else:
            full_data = data

    print (full_data.count())
    full_data.repartition(1).write.format("libsvm").save(rdd_name)
