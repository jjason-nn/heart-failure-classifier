from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Initialize Spark Session
spark = SparkSession.builder.appName("Heart Failure Classifier").getOrCreate()

#Loading the dataset
data = spark.read.csv("heart.csv", inferSchema=True, header=True)

#Data pipeline
categorical_cols = ["Sex","ChestPainType","RestingECG","ST_Slope"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]