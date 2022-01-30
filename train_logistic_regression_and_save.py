from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

from pyspark.ml.classification import LogisticRegression

from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import col
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator


conf = SparkConf()
conf.setMaster("local[*]").setAppName("CENG790-Project")
conf.set("spark.driver.memory", "50g")
conf.set("spark.driver.maxResultSize", "0")

spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

schema = StructType([StructField('features', VectorUDT(),False), StructField('label', DoubleType(),False)])

rescaledData = spark.read.schema(schema=schema).json("dataset_small_vectorized_binary.json")
# rescaledData = spark.read.schema(schema=schema).json("dataset_vectorized_binary.json")

# Class weightening by creating a weight column and multiplying the 
# label column(0 for turkish, 1 for non-turkish) by some weighting_constant.
weighting_constant = 9
rescaledData = rescaledData.withColumn('weight',  (col('label') * weighting_constant) + 1)
rescaledData.show(10)

(trainingData, testData) = rescaledData.select("label", "features", "weight").randomSplit([0.8, 0.2], seed=0)

# Logistic regression with default parameters and no class weighting
# lr = LogisticRegression(featuresCol="features", labelCol="label")

# Logistic regression with default parameters and class weighting.
lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="weight")

evaluator = BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction")

paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 1.0])\
    .build()

tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8,
                           parallelism=10)

# Fit the model
lrModel = tvs.fit(trainingData)

lrModel.bestModel.save("model_logistic_regression")


