from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.classification import LogisticRegressionModel

from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics


# Initialize spark.
conf = SparkConf()
conf.setMaster("local[*]").setAppName("CENG790-Project")
conf.set("spark.driver.memory", "50g")
conf.set("spark.driver.maxResultSize", "0")

spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# Read vector data.
# Note: Change dataset name for full dataset prediction.
schema = StructType([StructField('features', VectorUDT(),False), StructField('label', DoubleType(),False)])

# rescaledData = spark.read.schema(schema=schema).json("dataset_small_vectorized_binary.json")
rescaledData = spark.read.schema(schema=schema).json("dataset_vectorized_binary.json")
rescaledData.show(10)


# Split train and test dataset by using seed=0
(trainingData, testData) = rescaledData.select("label", "features").randomSplit([0.8, 0.2], seed=0)


# Logistic regression
lrModel = LogisticRegressionModel.load("model_logistic_regression")
print(lrModel)
predictedTestDataLR = lrModel.transform(testData)
metrics1 = MulticlassMetrics(predictedTestDataLR.select("prediction", "label").rdd.map(tuple))
print("LR Confusion Matrix: ")
print(metrics1.confusionMatrix().toArray())

predictedTestDataLR = predictedTestDataLR.withColumnRenamed("prediction", "prediction1")
predictedTestDataLR = predictedTestDataLR.withColumnRenamed("probability", "probability1")
predictedTestDataLR = predictedTestDataLR.withColumnRenamed("rawPrediction", "rawPrediction1")

# Random Forest
rfModel = RandomForestClassificationModel.load("model_random_forest")
print(rfModel)
predictedTestDataRF = rfModel.transform(predictedTestDataLR)
metrics2 = MulticlassMetrics(predictedTestDataRF.select("prediction", "label").rdd.map(tuple))
print("RF Confusion Matrix: ")
print(metrics2.confusionMatrix().toArray())

predictedTestDataRF = predictedTestDataRF.withColumnRenamed("prediction", "prediction2")
predictedTestDataRF = predictedTestDataRF.withColumnRenamed("probability", "probability2")
predictedTestDataRF = predictedTestDataRF.withColumnRenamed("rawPrediction", "rawPrediction2")

# Combine results
combinedTestData = predictedTestDataRF

combinedResultTestData = combinedTestData.withColumn('prediction', (combinedTestData.prediction1.cast('boolean') & combinedTestData.prediction2.cast('boolean')).cast('double'))\
    .withColumn('probability', combinedTestData.probability1)

metrics = MulticlassMetrics(combinedResultTestData.select("prediction", "label").rdd.map(tuple))
print("LR+RF Confusion Matrix: ")
print(metrics.confusionMatrix().toArray())
