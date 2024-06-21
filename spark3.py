from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when, hour, minute
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pysparkling import H2OContext
import h2o
from h2o.automl import H2OAutoML

# Initialize Spark session
spark = SparkSession.builder \
    .appName("DeliveryTimePrediction") \
    .master("spark://172.20.250.207:7077") \
    .getOrCreate()


# Initialize H2O context
h2o_context = H2OContext.getOrCreate()
# Load the dataset
df = spark.read.csv("/home/hduser_/Desktop/ETBDC_project/Data/ZomatoDataset1.csv", header=True, inferSchema=True)



# Handling missing values
# Filling numerical columns with mean
numerical_cols = ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]
for col_name in numerical_cols:
    mean_value = df.select(mean(col(col_name))).collect()[0][0]
    df = df.fillna({col_name: mean_value})

#Showing the original dataset.
print("Showing the original Dataset.")
df.printSchema()
df.show()

# Filling categorical columns with 'Unknown'
categorical_cols = ["Weather_conditions", "Road_traffic_density", "Festival", "City"]
for col_name in categorical_cols:
    df = df.fillna({col_name: "Unknown"})

# Convert categorical columns to numerical using one-hot encoding


indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_vec") for col in categorical_cols]

# Extract features from datetime fields
df = df.withColumn("Order_Hour", hour(col("Time_Orderd").cast("timestamp")))
df = df.withColumn("Order_Minute", minute(col("Time_Orderd").cast("timestamp")))

# Drop unnecessary columns
print("Dropping unnecessay columns:")
df = df.drop("ID", "Delivery_person_ID", "Order_Date", "Time_Orderd", "Time_Order_picked")

# Build the pipeline
pipeline = Pipeline(stages=indexers + encoders)
df_transformed = pipeline.fit(df).transform(df)

# Show the preprocessed data
df_transformed.show()



# Convert Spark DataFrame to H2O Frame
h2o_frame = h2o_context.asH2OFrame(df_transformed)

# Split the data into training and test sets
train, test = h2o_frame.split_frame(ratios=[0.8], seed=42)

# Define predictors and response variable
predictors = [col for col in h2o_frame.columns if col != "Time_taken (min)"]
response = "Time_taken (min)"

# Train models using H2O AutoML
print("Training models using automl:")
aml = H2OAutoML(max_runtime_secs=3600, seed=42)
aml.train(x=predictors, y=response, training_frame=train)

# Get the best model
best_model = aml.leader

# Make predictions on the test set
predictions = best_model.predict(test)

# Evaluate model performance on the test set
performance = best_model.model_performance(test)

# Printing the performance metrics
print("Model Performance Metrics on Test Set:")
print(f"RMSE: {performance.rmse()}")

# Convert predictions to Spark DataFrame
predictions_spark = h2o_context.asSparkFrame(predictions)

# Show predictions
print("Showing predictions:")
predictions_spark.show()



# Plot feature importance
#best_model.varimp_plot()

h2o_context.stop()

spark.stop()

