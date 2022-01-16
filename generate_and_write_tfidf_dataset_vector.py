from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer


conf = SparkConf()
conf.setMaster("local[*]").setAppName("CENG790-Project")
conf.set("spark.driver.memory", "15g")

spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# Reads from the dataset_json.json file, 
df_json = spark.read.json("dataset.json")

current_df = df_json

# TFIDF
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(current_df)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
featurizedData = hashingTF.transform(wordsData)
# alternatively, CountVectorizer can also be used to get term frequency vectors

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledDataWithLang = idfModel.transform(featurizedData)

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="lang", outputCol="label")
rescaledData = indexer.fit(rescaledDataWithLang).transform(rescaledDataWithLang)

print(rescaledData.count())


rescaledData.select("features", "label").write.mode("overwrite").json("dataset_vectorized.json")