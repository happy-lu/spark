from __future__ import print_function

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
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

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
    print("model detail\n:" + rfModel.toDebugString)
    # $example off$

    spark.stop()


def read_csv(spark, file_name):
    sql_context = SQLContext(spark)

    customSchema = StructType([
        StructField("date", StringType(), True)])

    df = sql_context.read.format('com.databricks.spark.csv').options(header='true', format="string").load(
        file_name)

    df.show()

    labelIndexer = StringIndexer(inputCol="failure", outputCol="indexedLabel").fit(data)
    input_cols = df.columns
    input_cols.remove("failure")
    assembler = VectorAssembler(
        inputCols=input_cols,
        outputCol="features")

    output = assembler.transform(df)
    output.select("features", "clicked").show(truncate=False)
    return output


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("RandomForestClassifierExample") \
        .getOrCreate()

    # $example on$
    # Load and parse the data file, converting it to a DataFrame.
    # data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
    data = read_csv(spark, "E://mldata//hard-disk-2016-q1-data//2016-01-11.csv")

    # exec_method(spark, data)
