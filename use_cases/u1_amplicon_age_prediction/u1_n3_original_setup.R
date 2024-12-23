# Load necessary packages
library(randomForest)
library(reticulate)
library(Metrics)  # For RMSE
library(caret)    # For R-squared

# Set seed for reproducibility
set.seed(123)

# Load pandas via reticulate
pd <- import("pandas")

# Read training and testing data
train_data <- pd$read_pickle("data_splits/train_val.pkl")
test_data <- pd$read_pickle("data_splits/test.pkl")

# Convert to data frames
train_df <- as.data.frame(train_data)
test_df <- as.data.frame(test_data)

# Extract columns starting with 'F' (features) and the target 'age_months'
predictor_cols <- grep("^F", names(train_df), value = TRUE)
train_predictors <- train_df[, predictor_cols]
train_target <- train_df$age_months

test_predictors <- test_df[, predictor_cols]
test_target <- test_df$age_months

# Convert absolute abundances to relative abundances
train_predictors_rel <- train_predictors / rowSums(train_predictors)
test_predictors_rel <- test_predictors / rowSums(test_predictors)

# Combine predictors and target for training
train_set <- data.frame(train_predictors_rel, age_months = train_target)

# Determine the number of predictors
p <- ncol(train_predictors_rel)

# Train Random Forest regression model
rf_model <- randomForest(age_months ~ ., data = train_set, ntree = 10000, mtry = round(p/3))

# Predictions on training data
train_preds <- predict(rf_model, train_predictors_rel)

# Predictions on testing data
test_preds <- predict(rf_model, test_predictors_rel)

# Calculate R² and RMSE for training data
train_r2 <- R2(train_preds, train_target)
train_rmse <- rmse(train_target, train_preds)

# Calculate R² and RMSE for testing data
test_r2 <- R2(test_preds, test_target)
test_rmse <- rmse(test_target, test_preds)

# Display performance metrics
cat("Training R-squared:", train_r2, "\n")
cat("Training RMSE:", train_rmse, "\n")
cat("Testing R-squared:", test_r2, "\n")
cat("Testing RMSE:", test_rmse, "\n")
