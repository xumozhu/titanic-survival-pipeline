
# ==== STEP 1: LOAD AND EXPLORE THE TITANIC DATASET ====

# Read the Titanic dataset and treat empty strings as missing values.
train_df <- read.csv("src/data/train.csv", na.strings = c("", "NA"))
# Display the first five rows of the dataset.
cat("First Five Rows:\n")
print(head(train_df, 5))

# Display the number of rows and columns in the dataset.
cat("\nDataset Shape:\n")
print(dim(train_df))

# Display the structure and data types of the dataset.
cat("\nDataset Structure:\n")
str(train_df)

# ==== STEP 2: CHECK MISSING VALUES ====

# Count the missing values in each column of the training dataset.
missing_counts <- colSums(is.na(train_df))

# Display the number of missing values for each column.
cat("\nMissing Values by Column:\n")
print(missing_counts)

# ==== STEP 3: SPLIT THE DATA INTO TRAINING AND VALIDATION SETS ====

# Set a random seed to make the data split reproducible.
set.seed(42)

# Separate passenger indices based on their survival outcomes.
survived_indices <- which(train_df$Survived == 1)
not_survived_indices <- which(train_df$Survived == 0)

# Randomly select approximately 80% of each survival group for training.
train_survived <- sample(
  survived_indices,
  size = floor(length(survived_indices) * 0.8)
)

train_not_survived <- sample(
  not_survived_indices,
  size = floor(length(not_survived_indices) * 0.8)
)

# Combine the selected indices to create the training set.
train_indices <- sort(c(train_survived, train_not_survived))

# Use the remaining passengers for validation.
val_indices <- setdiff(seq_len(nrow(train_df)), train_indices)

# Create the training and validation data frames.
train_data <- train_df[train_indices, ]
val_data <- train_df[val_indices, ]

# Display the dimensions of the training and validation sets.
cat("\nTraining Set Shape:\n")
print(dim(train_data))

cat("\nValidation Set Shape:\n")
print(dim(val_data))

# Check the survival proportions in both sets.
cat("\nTraining Survival Rate:\n")
print(mean(train_data$Survived))

cat("\nValidation Survival Rate:\n")
print(mean(val_data$Survived))

# ==== STEP 4: HANDLE MISSING VALUES WITHOUT DATA LEAKAGE ====

# Calculate the median passenger age using only the training set.
train_age_median <- median(train_data$Age, na.rm = TRUE)

# Replace missing ages in both datasets using the training median.
train_data$Age[is.na(train_data$Age)] <- train_age_median
val_data$Age[is.na(val_data$Age)] <- train_age_median

# Find the most common embarkation port using only the training set.
train_embarked_mode <- names(
  sort(table(train_data$Embarked), decreasing = TRUE)
)[1]

# Replace missing embarkation ports using the training mode.
train_data$Embarked[is.na(train_data$Embarked)] <- train_embarked_mode
val_data$Embarked[is.na(val_data$Embarked)] <- train_embarked_mode

# Display the values used to fill missing data.
cat("\nTraining Age Median Used:\n")
print(train_age_median)

cat("\nTraining Embarked Mode Used:\n")
print(train_embarked_mode)

# Verify that the model features no longer contain missing values.
cat("\nTraining Age Missing:\n")
print(sum(is.na(train_data$Age)))

cat("\nValidation Age Missing:\n")
print(sum(is.na(val_data$Age)))

cat("\nTraining Embarked Missing:\n")
print(sum(is.na(train_data$Embarked)))

cat("\nValidation Embarked Missing:\n")
print(sum(is.na(val_data$Embarked)))

# ==== STEP 5: SELECT FEATURES FOR LOGISTIC REGRESSION ====

# Select the same seven predictor variables used in the Python model.
selected_features <- c(
  "Pclass", "Sex", "Age", "SibSp",
  "Parch", "Fare", "Embarked"
)

# Keep the survival outcome and selected predictors for training.
train_model_data <- train_data[, c("Survived", selected_features)]

# Keep the same predictors in the validation dataset.
val_model_data <- val_data[, c("Survived", selected_features)]

# Convert categorical variables into factors for statistical modeling.
train_model_data$Sex <- factor(train_model_data$Sex)
train_model_data$Embarked <- factor(train_model_data$Embarked)

# Use the training categories for validation to ensure consistency.
val_model_data$Sex <- factor(
  val_model_data$Sex,
  levels = levels(train_model_data$Sex)
)

val_model_data$Embarked <- factor(
  val_model_data$Embarked,
  levels = levels(train_model_data$Embarked)
)

# Display the selected model features.
cat("\nSelected Features:\n")
print(selected_features)

# Display the training and validation dimensions.
cat("\nTraining Model Data Shape:\n")
print(dim(train_model_data))

cat("\nValidation Model Data Shape:\n")
print(dim(val_model_data))


# ==== STEP 6: TRAIN THE R LOGISTIC REGRESSION MODEL ====

# Train a logistic regression model using the selected Titanic features.
r_model <- glm(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = train_model_data,
  family = binomial(link = "logit")
)

# Confirm that model training has completed.
cat("\nR Logistic Regression model trained successfully!\n")

# Display the estimated model coefficients.
cat("\nModel Coefficients:\n")
print(coef(r_model))

# ==== STEP 7: PREDICT AND EVALUATE VALIDATION ACCURACY ====

# Predict survival probabilities for passengers in the validation set.
val_probabilities <- predict(
  r_model,
  newdata = val_model_data,
  type = "response"
)

# Convert survival probabilities into binary predictions using 0.5 as the threshold.
val_predictions <- ifelse(val_probabilities >= 0.5, 1, 0)

# Compare the predicted outcomes with the actual survival labels.
correct_predictions <- sum(
  val_predictions == val_model_data$Survived
)

# Calculate the proportion of correctly classified passengers.
validation_accuracy <- mean(
  val_predictions == val_model_data$Survived
)

# Display the first ten predicted and actual survival outcomes.
cat("\nFirst 10 Validation Predictions:\n")
print(head(val_predictions, 10))

cat("\nFirst 10 Actual Survival Outcomes:\n")
print(head(val_model_data$Survived, 10))

# Display the validation accuracy as a percentage.
cat("\nR Validation Accuracy:\n")
cat(sprintf("%.2f%%\n", validation_accuracy * 100))

# Display the number of correct predictions.
cat("\nCorrect Predictions:\n")
cat(correct_predictions, "out of", nrow(val_model_data), "\n")

# ==== STEP 8: PREDICT SURVIVAL FOR THE KAGGLE TEST DATASET ====

# Load the Kaggle test dataset and recognize empty values as missing.
test_df <- read.csv(
  "src/data/test.csv",
  na.strings = c("", "NA")
)

# Display the number of passengers and features in the test dataset.
cat("\nKaggle Test Dataset Shape:\n")
print(dim(test_df))

# Calculate the median fare using only the training dataset.
train_fare_median <- median(train_data$Fare, na.rm = TRUE)

# Fill missing ages and fares using training-data statistics.
test_df$Age[is.na(test_df$Age)] <- train_age_median
test_df$Fare[is.na(test_df$Fare)] <- train_fare_median

# Fill missing embarkation ports using the training-data mode.
test_df$Embarked[is.na(test_df$Embarked)] <- train_embarked_mode

# Select the same predictor variables used to train the model.
test_model_data <- test_df[, selected_features]

# Apply the same categorical levels used by the training dataset.
test_model_data$Sex <- factor(
  test_model_data$Sex,
  levels = levels(train_model_data$Sex)
)

test_model_data$Embarked <- factor(
  test_model_data$Embarked,
  levels = levels(train_model_data$Embarked)
)

# Check that the selected test features contain no missing values.
cat("\nTest Model Missing Values:\n")
print(colSums(is.na(test_model_data)))

# Predict survival probabilities for all Kaggle test passengers.
test_probabilities <- predict(
  r_model,
  newdata = test_model_data,
  type = "response"
)

# Convert predicted probabilities into binary survival outcomes.
test_predictions <- ifelse(test_probabilities >= 0.5, 1, 0)

# Display the total number of predictions and predicted survivors.
cat("\nTotal Test Predictions:\n")
print(length(test_predictions))

cat("\nPredicted Survivors:\n")
print(sum(test_predictions == 1))

cat("\nPredicted Non-Survivors:\n")
print(sum(test_predictions == 0))

# Prepare the Kaggle submission with passenger IDs and survival predictions.
submission_df <- data.frame(
  PassengerId = test_df$PassengerId,
  Survived = test_predictions
)

# Save the R predictions to a separate CSV file.
write.csv(
  submission_df,
  "src/data/titanic_submission_r.csv",
  row.names = FALSE
)

# Display the first five rows and final dimensions of the submission.
cat("\nFirst Five R Submission Rows:\n")
print(head(submission_df, 5))

cat("\nR Submission Shape:\n")
print(dim(submission_df))

cat("\nR submission saved to: src/data/titanic_submission_r.csv\n")
