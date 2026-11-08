
# =============================================================================
# MLDS 400 - Homework 3: Titanic Survival Prediction
# Python Logistic Regression Model
# =============================================================================


# =============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# =============================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =============================================================================
# STEP 2: LOAD AND EXPLORE THE TITANIC DATASET
# =============================================================================

# Load the Titanic training dataset into a pandas DataFrame.
train_df = pd.read_csv("src/data/train.csv")

# Display the first five rows to understand the dataset structure.
print("\nFirst Five Rows:")
print(train_df.head())

# Display the total number of rows and columns.
print("\nDataset Shape:")
print(train_df.shape)

# Display column names, data types, and non-null counts.
print("\nDataset Information:")
train_df.info()


# =============================================================================
# STEP 3: IDENTIFY MISSING VALUES
# =============================================================================

# Count the missing values in each column.
missing_values = train_df.isna().sum()

# Display the missing value counts.
print("\nMissing Values by Column:")
print(missing_values)

# Calculate the overall median age for initial data exploration.
# This value is not used for model preprocessing to avoid data leakage.
age_median = train_df["Age"].median()

# Display the overall median age.
print("\nOverall Median Age:", age_median)


# =============================================================================
# STEP 4: SPLIT DATA INTO TRAINING AND VALIDATION SETS
# =============================================================================

# Separate the target variable from the passenger features.
X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

# Split the dataset into 80% training and 20% validation.
# Stratification maintains a similar survival ratio in both subsets.
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the dimensions of the training and validation datasets.
print("\nTraining Set Shape:", X_train.shape)
print("Validation Set Shape:", X_val.shape)


# =============================================================================
# STEP 5: HANDLE MISSING AGE VALUES (MEDIAN IMPUTATION)
# =============================================================================

# Create independent copies to avoid modifying the original split data.
X_train = X_train.copy()
X_val = X_val.copy()

# Calculate the median age using only the training subset.
train_age_median = X_train["Age"].median()

# Fill missing ages in both subsets using the training median.
X_train["Age"] = X_train["Age"].fillna(train_age_median)
X_val["Age"] = X_val["Age"].fillna(train_age_median)

# Verify that all missing age values have been filled.
print("\nTraining Age Missing:", X_train["Age"].isna().sum())
print("Validation Age Missing:", X_val["Age"].isna().sum())
print("Training Age Median Used:", train_age_median)


# =============================================================================
# STEP 6: HANDLE MISSING EMBARKED VALUES (MODE IMPUTATION)
# =============================================================================

# Identify the most frequent embarkation port in the training subset.
train_embarked_mode = X_train["Embarked"].mode()[0]

# Fill missing embarkation ports using the training mode.
X_train["Embarked"] = X_train["Embarked"].fillna(train_embarked_mode)
X_val["Embarked"] = X_val["Embarked"].fillna(train_embarked_mode)

# Verify that all missing embarkation values have been filled.
print("\nTraining Embarked Missing:", X_train["Embarked"].isna().sum())
print("Validation Embarked Missing:", X_val["Embarked"].isna().sum())
print("Training Embarked Mode Used:", train_embarked_mode)


# =============================================================================
# STEP 7: SELECT FEATURES FOR LOGISTIC REGRESSION
# =============================================================================

# Select the passenger features used to predict survival.
selected_features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

# Keep only the selected features in both datasets.
X_train = X_train[selected_features].copy()
X_val = X_val[selected_features].copy()

# Display the selected features and dataset dimensions.
print("\nSelected Features:", selected_features)
print("Training Features Shape:", X_train.shape)
print("Validation Features Shape:", X_val.shape)


# =============================================================================
# STEP 8: APPLY ONE-HOT ENCODING TO CATEGORICAL FEATURES
# =============================================================================

# Identify categorical columns that need numeric encoding.
categorical_features = ["Sex", "Embarked"]

# Create an encoder and remove one reference category per feature.
# Ignore unknown categories when processing future observations.
encoder = OneHotEncoder(
    drop="first",
    handle_unknown="ignore",
    sparse_output=False
)

# Learn the encoding rules from training data and transform it.
X_train_encoded = encoder.fit_transform(
    X_train[categorical_features]
)

# Apply the same encoding rules to validation data.
X_val_encoded = encoder.transform(
    X_val[categorical_features]
)

# Display the new feature names and encoded dataset dimensions.
print("\nEncoded Feature Names:")
print(encoder.get_feature_names_out(categorical_features))

print("\nEncoded Training Shape:", X_train_encoded.shape)
print("Encoded Validation Shape:", X_val_encoded.shape)


# =============================================================================
# STEP 9: COMBINE NUMERIC AND ENCODED FEATURES
# =============================================================================

# Identify the original numeric features.
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

# Retrieve the names of the one-hot encoded columns.
encoded_feature_names = encoder.get_feature_names_out(
    categorical_features
)

# Convert encoded training data into a DataFrame with matching indexes.
train_encoded_df = pd.DataFrame(
    X_train_encoded,
    columns=encoded_feature_names,
    index=X_train.index
)

# Convert encoded validation data into a DataFrame with matching indexes.
val_encoded_df = pd.DataFrame(
    X_val_encoded,
    columns=encoded_feature_names,
    index=X_val.index
)

# Combine numeric and encoded training features horizontally.
X_train_final = pd.concat(
    [X_train[numeric_features], train_encoded_df],
    axis=1
)

# Combine numeric and encoded validation features horizontally.
X_val_final = pd.concat(
    [X_val[numeric_features], val_encoded_df],
    axis=1
)

# Verify the final dimensions and confirm there are no missing values.
print("\nFinal Training Shape:", X_train_final.shape)
print("Final Validation Shape:", X_val_final.shape)

print(
    "Final Training Missing Values:",
    X_train_final.isna().sum().sum()
)

print(
    "Final Validation Missing Values:",
    X_val_final.isna().sum().sum()
)


# =============================================================================
# STEP 10: TRAIN THE LOGISTIC REGRESSION MODEL
# =============================================================================

# Initialize the logistic regression classifier.
# Allow up to 1000 iterations for the optimization algorithm.
model = LogisticRegression(max_iter=1000)

# Train the model using processed features and actual survival labels.
model.fit(X_train_final, y_train)
print("\nLogistic Regression model trained successfully!")

# =============================================================================
# STEP 11: MAKE VALIDATION PREDICTIONS
# =============================================================================

# Predict survival outcomes for passengers in the validation dataset.
y_pred = model.predict(X_val_final)

# Display the first 10 predicted survival outcomes.
print("\nFirst 10 Validation Predictions:")
print(y_pred[:10])

# Display the first 10 actual survival outcomes for comparison.
print("\nFirst 10 Actual Survival Outcomes:")
print(y_val.iloc[:10].to_numpy())

# =============================================================================
# STEP 12: EVALUATE MODEL ACCURACY
# =============================================================================

# Calculate validation accuracy by comparing predictions with actual labels.
validation_accuracy = accuracy_score(y_val, y_pred)

# Calculate the number of correctly predicted passengers.
correct_predictions = (y_val.to_numpy() == y_pred).sum()

# Display the validation accuracy as a percentage.
print("\nValidation Accuracy:", f"{validation_accuracy:.2%}")

# Display the number of correct predictions and total passengers.
print(
    "Correct Predictions:",
    correct_predictions,
    "out of",
    len(y_val)
)

# =============================================================================
# STEP 13: LOAD AND PREPARE THE KAGGLE TEST DATASET
# =============================================================================

# Load the official Kaggle test dataset containing unseen passengers.
test_df = pd.read_csv("src/data/test.csv")

# Display the dimensions of the test dataset.
print("\nKaggle Test Dataset Shape:", test_df.shape)

# Display the first five rows to inspect the test data.
print("\nFirst Five Rows of Kaggle Test Data:",test_df.head())

# Count missing values in each column before preprocessing.
print("\nKaggle Test Missing Values:",test_df.isna().sum())

# =============================================================================
# STEP 13B: PREPROCESS THE KAGGLE TEST DATA
# =============================================================================

# Create a copy to preserve the original Kaggle test dataset.
X_test = test_df.copy()

# Calculate the fare median using only the training subset.
train_fare_median = X_train["Fare"].median()

# Fill missing ages using the previously learned training median.
X_test["Age"] = X_test["Age"].fillna(train_age_median)

# Fill missing fares using the training subset's fare median.
X_test["Fare"] = X_test["Fare"].fillna(train_fare_median)

# Fill any missing embarkation ports using the training mode.
X_test["Embarked"] = X_test["Embarked"].fillna(train_embarked_mode)

# Keep the same seven features used during model training.
X_test = X_test[selected_features].copy()

# Verify the processed test dataset and the fare median used.
print("\nTraining Fare Median Used:", train_fare_median)
print("Test Features Shape:", X_test.shape)
print("Test Missing Values:", X_test.isna().sum().sum())

# =============================================================================
# STEP 13C: ENCODE AND COMBINE KAGGLE TEST FEATURES
# =============================================================================

# Apply the existing one-hot encoder to the test categorical features.
# Do not fit the encoder again because it was trained on training data.
X_test_encoded = encoder.transform(
    X_test[categorical_features]
)

# Convert encoded test features into a DataFrame with matching row indexes.
test_encoded_df = pd.DataFrame(
    X_test_encoded,
    columns=encoded_feature_names,
    index=X_test.index
)

# Combine the original numeric features with the encoded categorical features.
X_test_final = pd.concat(
    [X_test[numeric_features], test_encoded_df],
    axis=1
)

# Match the training feature names and their exact column order.
X_test_final = X_test_final[X_train_final.columns]

# Verify that the final test data is ready for model predictions.
print("\nFinal Test Shape:", X_test_final.shape)
print("Final Test Missing Values:", X_test_final.isna().sum().sum())
print("Test Features Match Training:", X_test_final.columns.equals(X_train_final.columns))

# =============================================================================
# STEP 14: PREDICT SURVIVAL ON THE KAGGLE TEST DATASET
# =============================================================================

# Predict survival outcomes for all passengers in the Kaggle test dataset.
test_predictions = model.predict(X_test_final)

# Display the first 10 predicted survival outcomes.
print("\nFirst 10 Kaggle Test Predictions:")
print(test_predictions[:10])

# Count how many passengers are predicted to survive.
predicted_survivors = (test_predictions == 1).sum()

# Count how many passengers are predicted not to survive.
predicted_non_survivors = (test_predictions == 0).sum()

# Display a summary of the predictions.
print("\nTotal Test Predictions:", len(test_predictions))
print("Predicted Survivors:", predicted_survivors)
print("Predicted Non-Survivors:", predicted_non_survivors)

# =============================================================================
# STEP 15: SAVE KAGGLE TEST PREDICTIONS
# =============================================================================
# Create a submission DataFrame with passenger IDs and predicted survival.
submission_df = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_predictions
})

# Save the predictions as a CSV file without including the DataFrame index.
submission_df.to_csv(
    "src/data/titanic_submission.csv",
    index=False
)

# Display the output file location and the first five predictions.
print("\nKaggle submission saved to: src/data/titanic_submission.csv")
print("\nFirst Five Submission Rows:")
print(submission_df.head())

# Confirm the number of rows and columns in the saved submission.
print("\nSubmission Shape:", submission_df.shape)
