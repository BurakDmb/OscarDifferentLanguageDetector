from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.classification import MultilayerPerceptronClassificationModel

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


# MLP
mlpModel = MultilayerPerceptronClassificationModel.load("model_mlp")

print(mlpModel)
print(mlpModel.getLayers())
print(mlpModel.getStepSize())

predictedTestData = mlpModel.transform(testData)

# evaluator = MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
#evaluator = BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction").setMetricName("areaUnderROC")
# evalResults = evaluator.evaluate(predictedTestData)
# print(evalResults)



metrics = MulticlassMetrics(predictedTestData.select("prediction", "label").rdd.map(tuple))
print(metrics.confusionMatrix().toArray())
