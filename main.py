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
categorical_cols = ["Sex","ChestPainType","RestingECG", "ExerciseAngina", "ST_Slope"]

# Transforming Categorical Columns to Numerical
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]

# Creating pipline for indexing categorical columns
pipeline = Pipeline(stages=indexers)

# Fit and Transform data
data_transformed = pipeline.fit(data).transform(data)

# data after transform
print("transformed data\n")
data_transformed.select(
    "Sex", "Sex_index", 
    "ChestPainType", "ChestPainType_index", 
    "RestingECG", "RestingECG_index", 
    "ST_Slope", "ST_Slope_index",
    "ExerciseAngina", "ExerciseAngina_index"
    ).show(5)

spark.stop()