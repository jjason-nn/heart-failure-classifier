from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Initialize Spark Session
spark = SparkSession.builder.appName("Heart Failure Classifier").getOrCreate()

#Loading the dataset
data = spark.read.csv("heart.csv", inferSchema=True, header=True)

#Data pipeline
categorical_cols = ["Sex","ChestPainType","RestingECG", "ExerciseAngina", "ST_Slope"]

# Transforming Categorical Columns to Numerical
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]

#Assemble features into a single vector
assembler = VectorAssembler(
    inputCols = ["Age", 
                 "RestingBP",
                 "Cholesterol",
                 "FastingBS",
                 "MaxHR",
                 "Oldpeak"] + [col +"_index" for col in categorical_cols[:-1]],
    outputCol = "features"
)

#Scale the features
scaler = StandardScaler(inputCol = "features", outputCol = "scaledFeatures", withStd = True, withMean = False)

#Logistic Regression Model
ml = RandomForestClassifier(featuresCol = "scaledFeatures", labelCol = "HeartDisease")

# Creating pipline
pipeline = Pipeline(stages=indexers + [assembler, scaler, ml])

#Split the data
train_data, test_data = data.randomSplit([0.8, 0.2], seed = 42)

# Train the model
model = pipeline.fit(train_data)

#Make predictions
predictions = model.transform(test_data)
predictions.select("HeartDisease", "scaledFeatures", "prediction", "probability").show(10)

# ================================================================ Performance Metrics ================================================================

#Area under ROC curve
evaluator = BinaryClassificationEvaluator(labelCol = "HeartDisease", metricName = "areaUnderROC")
roc_auc = evaluator.evaluate(predictions)
print(f"Area under curve for ROC graph: {roc_auc}")

# Confusion Matrix
confusion_matrix = predictions.groupBy("HeartDisease", "prediction"
                                       ).count(
                                        ).orderBy("HeartDisease", "prediction")
confusion_matrix.show(10)

# Accuracy
accuracy = predictions.filter(predictions.HeartDisease == predictions.prediction).count() / predictions.count()
print(f"Accuracy: {accuracy}")

# Precision, Recall, F1-Score
true_pos = predictions.filter((predictions.HeartDisease == 1) & (predictions.prediction == 1)).count()
false_pos = predictions.filter((predictions.HeartDisease == 0) & (predictions.prediction == 1)).count()
true_neg = predictions.filter((predictions.HeartDisease == 0) & (predictions.prediction == 0)).count()
false_neg = predictions.filter((predictions.HeartDisease == 1) & (predictions.prediction == 0)).count()

precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Score: {f_score}")
spark.stop()