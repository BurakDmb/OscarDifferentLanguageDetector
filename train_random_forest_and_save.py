from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import col
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

conf = SparkConf()
conf.setMaster("local[*]").setAppName("CENG790-Project")
conf.set("spark.driver.memory", "50g")
# conf.set("spark.executor.memory", "2g")
# conf.set("spark.driver.maxResultSize", "0")

spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")


schema = StructType([StructField('features', VectorUDT(),False), StructField('label', DoubleType(),False)])

# rescaledData = spark.read.schema(schema=schema).json("dataset_small_vectorized_binary.json")
rescaledData = spark.read.schema(schema=schema).json("dataset_vectorized_binary.json")


# Class weightening by creating a weight column and multiplying the 
# label column(0 for turkish, 1 for non-turkish) by some weighting_constant.

weighting_constant = 9
rescaledData = rescaledData.withColumn('weight',  (col('label') * weighting_constant) + 1)
rescaledData.show(10)

(trainingData, testData) = rescaledData.select("label", "features", "weight").randomSplit([0.8, 0.2], seed=0)

# RF classifier with default parameters - no class weighting
# rf = RandomForestClassifier(featuresCol="features", labelCol="label")

# RF classifier with default parameters - with class weighting
rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="weight")

evaluator = BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction")

paramGrid = ParamGridBuilder()\
    .addGrid(rf.maxBins, [8, 16])\
    .addGrid(rf.maxDepth, [3, 5])\
    .addGrid(rf.numTrees, [2, 10, 20])\
    .addGrid(rf.subsamplingRate, [0.1, 0.2])\
    .build()

tvs = TrainValidationSplit(estimator=rf,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8,
                           parallelism=10)

# Fit the model
rfModel = tvs.fit(trainingData)

rfModel.bestModel.save("model_random_forest")


