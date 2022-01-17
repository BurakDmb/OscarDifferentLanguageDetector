from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

from pyspark.ml.classification import LogisticRegression

from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import StructType, StructField, DoubleType


conf = SparkConf()
conf.setMaster("local[*]").setAppName("CENG790-Project")
conf.set("spark.driver.memory", "50g")
conf.set("spark.driver.maxResultSize", "0")

spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

schema = StructType([StructField('features', VectorUDT(),False), StructField('label', DoubleType(),False)])

vectorizedData = spark.read.schema(schema=schema).json("dataset_vectorized.json")


vectorizedData = vectorizedData.withColumnRenamed("label", "label_raw")
vectorizedData = vectorizedData.withColumn('label', (vectorizedData.label_raw.cast('boolean')).cast('double'))

vectorizedData.select("features", "label").write.mode("overwrite").json("dataset_vectorized_binary.json")
