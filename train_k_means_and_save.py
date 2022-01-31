from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.clustering import KMeans

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

# Prepare an equalized dataset
trData = trainingData.filter("label == 0")
frData = trainingData.filter("label == 1")

# Scale down the TR documents
trData = trData.sample(withReplacement=False, fraction=(1.0 * frData.count())/trData.count(), seed = 0 )

equalizedData = trData.union(frData)

# KMeans clustering with 2 clusters
kmeans = KMeans().setK(2).setSeed(0).setMaxIter(20).setInitSteps(2).setInitMode("random")

# Train with all training data
kMeansModel = kmeans.fit(trainingData)
kMeansModel.save("model_k_means_train")

# Train with equalized data
kMeansModelEq = kmeans.fit(equalizedData)
kMeansModelEq.save("model_k_means_eq")

