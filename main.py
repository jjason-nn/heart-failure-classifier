from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params


# Initialize Spark Session
spark = SparkSession.builder.appName("Heart Failure Classifier").getOrCreate()

# Loading the dataset
data = spark.read.csv("heart.csv", inferSchema=True, header=True)

# ================================================================ Data Preprocessing ================================================================
# Custom Transformer class to drop rows with NaN values
class NaNDroppingTransformer(Transformer):
    def __init__(self):
        super(NaNDroppingTransformer, self).__init__()
    
    def _transform(self, dataset):
        return dataset.na.drop()

# Data pipeline
categorical_cols = ["Sex","ChestPainType","RestingECG", "ExerciseAngina", "ST_Slope"]

# Transforming Categorical Columns to Numerical
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]

# ================================================================ Feature Engineering ================================================================

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
# Scale the features
scaler = StandardScaler(inputCol = "features", outputCol = "scaledFeatures", withStd = True, withMean = False)


# ================================================================ Model Training ================================================================
# Random Forest Classifier Model
ml = RandomForestClassifier(featuresCol = "scaledFeatures", labelCol = "HeartDisease")

# Creating pipline
pipeline = Pipeline(stages = [NaNDroppingTransformer()] + indexers + [assembler, scaler, ml])

# Split the data  80% for training and 20% for testing 
train_data, test_data = data.randomSplit([0.8, 0.2], seed = 42)

# Train the model
model = pipeline.fit(train_data)

#Make predictions
transformed_test_data = model.transform(test_data)
transformed_test_data.select("HeartDisease", "scaledFeatures", "prediction", "probability").show(10)
# ================================================================ Performance Metric/Evaluation ================================================================

# Area under ROC curve
evaluator = BinaryClassificationEvaluator(labelCol = "HeartDisease", metricName = "areaUnderROC")
roc_auc = evaluator.evaluate(transformed_test_data)
print(f"Area under curve for ROC graph: {roc_auc}")

# Confusion Matrix
confusion_matrix = transformed_test_data.groupBy("HeartDisease", "prediction"
                                       ).count(
                                        ).orderBy("HeartDisease", "prediction")
confusion_matrix.show(10)

# Accuracy
accuracy = transformed_test_data.filter(transformed_test_data.HeartDisease == transformed_test_data.prediction).count() / transformed_test_data.count()
print(f"Accuracy: {accuracy}")

# Precision, Recall, F1-Score
true_pos = transformed_test_data.filter((transformed_test_data.HeartDisease == 1) & (transformed_test_data.prediction == 1)).count()
false_pos = transformed_test_data.filter((transformed_test_data.HeartDisease == 0) & (transformed_test_data.prediction == 1)).count()
true_neg = transformed_test_data.filter((transformed_test_data.HeartDisease == 0) & (transformed_test_data.prediction == 0)).count()
false_neg = transformed_test_data.filter((transformed_test_data.HeartDisease == 1) & (transformed_test_data.prediction == 0)).count()

precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Score: {f_score}")

spark.stop()