from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassificationModel

from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics


conf = SparkConf()
conf.setMaster("local[*]").setAppName("CENG790-Project")
conf.set("spark.driver.memory", "50g")
conf.set("spark.driver.maxResultSize", "0")

spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

schema = StructType([StructField('features', VectorUDT(),False), StructField('label', DoubleType(),False)])

# rescaledData = spark.read.schema(schema=schema).json("dataset_small_vectorized_binary.json")
rescaledData = spark.read.schema(schema=schema).json("dataset_vectorized_binary.json")
rescaledData.show(10)

(trainingData, testData) = rescaledData.select("label", "features").randomSplit([0.8, 0.2], seed=0)

# Loading model, predicting test data and calculate evaluation metric by using MulticlassClassificationEvaluator

# Random Forest
rfModel = RandomForestClassificationModel.load("model_random_forest")

print(rfModel)
print(rfModel.getMaxBins())
print(rfModel.getMaxDepth())
print(rfModel.getNumTrees)
print(rfModel.getSubsamplingRate())

predictedTestData = rfModel.transform(testData)

evaluator = BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction").setMetricName("areaUnderROC")
evalResults = evaluator.evaluate(predictedTestData)
print(evalResults)



metrics = MulticlassMetrics(predictedTestData.select("prediction", "label").rdd.map(tuple))
print(metrics.confusionMatrix().toArray())
