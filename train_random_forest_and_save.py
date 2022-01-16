from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import StructType, StructField, DoubleType

conf = SparkConf()
conf.setMaster("local[*]").setAppName("CENG790-Project")
conf.set("spark.driver.memory", "50g")
conf.set("spark.driver.maxResultSize", "0")

spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")



schema = StructType([StructField('features', VectorUDT(),False), StructField('label', DoubleType(),False)])

rescaledData = spark.read.schema(schema=schema).json("dataset_small_rescaled.json")
rescaledData.show(10)

(trainingData, testData) = rescaledData.select("label", "features").randomSplit([0.8, 0.2], seed=0)

rf = RandomForestClassifier(featuresCol="features", labelCol="label", maxBins=25, maxDepth=4, impurity="entropy")

# Fit the model
rfModel = rf.fit(trainingData)

rfModel.save("model_random_forest")


