from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from sklearn.metrics import confusion_matrix

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

# Load model and transform
kMeansModel = KMeansModel.load("model_k_means_train")
print(kMeansModel)
predictedTestData = kMeansModel.transform(testData)

pred = predictedTestData.select("prediction").collect()
gt = predictedTestData.select("label").collect()

print("kMeansModel")
print(confusion_matrix(gt, pred))

# Same procedure for equalized data
kMeansModelEq = KMeansModel.load("model_k_means_eq")
print(kMeansModelEq)
predictedTestData = kMeansModelEq.transform(testData)

pred = predictedTestData.select("prediction").collect()
gt = predictedTestData.select("label").collect()

print("kMeansModelEq")
print(confusion_matrix(gt, pred))