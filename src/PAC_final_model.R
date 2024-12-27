# ====================================
# Setup and initialization
# ====================================

# Import required libraries for data manipulation, visualization, and modeling
library(skimr)
library(dplyr)
library(readr)
library(ggplot2)
library(lattice)
library(caret)
library(recipes)
library(vtreat)
library(xgboost)
library(Matrix)
library(tidyr)
library(tidyverse)
library(forcats)
library(janitor)

# Set random seed for reproducibility
set.seed(1031)

# ====================================
# Data loading and initial inspection
# ====================================

# Import the dataset
analysis_data = read.csv('data/analysis_data.csv', stringsAsFactors = TRUE)

# Examine data structure
str(analysis_data)

#===============================================
# Data Type Conversion
#===============================================

# Define columns that should be treated as factors
cols_to_factor <- c("contextual_relevance", "seasonality", "headline_power_words",
                    "headline_question", "headline_numbers")

# Convert specified columns to factors
analysis_data[cols_to_factor] <- lapply(analysis_data[cols_to_factor], as.factor)

#===============================================
# Data Quality Checks
#===============================================

# Check for duplicate IDs
analysis_data %>%
  group_by(id) %>%
  count() %>%
  filter(n==1) # Confirms no duplicates - all 4,000 rows have unique IDs

# Evaluate missing data
skim(analysis_data)

# ====================================
# Data validation and range checking
# ====================================

# Function to analyze correlations with target and handle out-of-range values
analyze_numeric <- function(analysis_df, column, min_val, max_val, cap) {
  # Check correlation with target variable (CTR)
  if(column != "CTR") {
    orig_cor <- cor(analysis_df[[column]], analysis_df$CTR, use = "complete.obs")
    cat(sprintf("\nCorrelation with CTR for %s: %.3f\n", column, orig_cor))
  }
  
  # Count values outside valid range
  out_range_analysis <- sum(analysis_df[[column]] < min_val | analysis_df[[column]] > max_val, na.rm = TRUE)
  cat(sprintf("\n%s: %d out of range values\n", column, out_range_analysis))
  
  # Cap values at min/max if specified
  if(cap) {
    analysis_df[[column]] <- ifelse(analysis_df[[column]] < min_val, min_val,
                                    ifelse(analysis_df[[column]] > max_val, max_val, 
                                           analysis_df[[column]]))
  }
  return(analysis_df)
}

# Define valid ranges and capping rules for numeric variables
numeric_vars <- list(
  list(col = "targeting_score", min = 1, max = 10, cap = FALSE),
  list(col = "visual_appeal", min = 1, max = 10, cap = FALSE), 
  list(col = "cta_strength", min = 1, max = 10, cap = FALSE),
  list(col = "brand_familiarity", min = 1, max = 10, cap = TRUE),
  list(col = "market_saturation", min = 1, max = 10, cap = TRUE),
  list(col = "body_keyword_density", min = 0, max = 1, cap = FALSE),
  list(col = "body_readability_score", min = 1, max = 100, cap = FALSE),
  list(col = "CTR", min = 0, max = 1, cap = TRUE)  # Target variable
)

# Define categorical variables
categorical_vars <- c("contextual_relevance", "age_group", "gender", "location",
                      "headline_power_words", "headline_question", "headline_numbers",
                      "seasonality")

# Apply range validation and capping to numeric variables
for(var in numeric_vars) {
  analysis_data <- analyze_numeric(analysis_data, var$col, var$min, var$max, var$cap)
}

# ====================================
# Range validation for special variables
# ====================================

# Check variables that should be between 0 and 1
zero_one_vars <- c("body_keyword_density", "body_readability_score")
for(var in zero_one_vars) {
  analysis_range <- any(analysis_data[[var]] > 1 | analysis_data[[var]] < 0, na.rm = TRUE)
  
  cat(sprintf("\n## %s range check:\n", var, analysis_range))
}

#===============================================
# Feature Engineering
#===============================================

# Create new features in analysis data
analysis_data <- analysis_data %>%
  mutate(
    # Average of key quality metrics
    content_score = (visual_appeal + targeting_score + cta_strength) / 3,
    # Binary indicator for peak hours
    peak_hour = ifelse(time_of_day %in% c("Morning", "Afternoon"), 1, 0)
  )

#===============================================
# Exploratory Data Analysis
#===============================================

# Data Structure Overview
str(analysis_data)

# Detailed Summary Statistics
skim(analysis_data)

# Create distribution plots for numeric variables
numeric_plot <- analysis_data %>% 
  select(-CTR, -id) %>%
  select_if(is.numeric) %>%
  pivot_longer(everything(), names_to = 'predictor', values_to = 'values') %>%
  ggplot(aes(x = values)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  facet_wrap(predictor ~ ., scales = 'free') +
  labs(title = "Distribution of Numeric Predictors") +
  theme_bw()

# Create distribution plots for categorical variables
categorical_plot <- analysis_data %>% 
  select(-CTR, -id) %>%
  select_if(is.factor) %>%
  pivot_longer(everything(), names_to = 'predictor', values_to = 'values') %>%
  ggplot(aes(x = values)) +
  geom_bar(fill = "lightgreen", color = "black") +
  facet_wrap(predictor ~ ., scales = 'free') +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Distribution of Categorical Predictors")

# Display plots
print(numeric_plot)
print(categorical_plot)

# CTR Analysis (Primary Metric)
summary(analysis_data$CTR)
# Check CTR distribution quantiles
quantile(analysis_data$CTR, probs = c(0, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1))

# Identify Key Relationships with CTR
# Get numeric columns
numeric_cols <- sapply(analysis_data, is.numeric)
# Calculate correlations with CTR
cor_with_ctr <- sapply(analysis_data[, numeric_cols], function(x) {
  cor(x, analysis_data$CTR, use = "complete.obs")
})
# Sort by absolute correlation value
cor_with_ctr <- sort(abs(cor_with_ctr), decreasing = TRUE)
print("Top correlations with CTR:")
print(head(cor_with_ctr, 10))

# Demographics Distribution
# Gender distribution
print(table(analysis_data$gender))
print(prop.table(table(analysis_data$gender)) * 100)

# Age distribution
print(table(analysis_data$age_group))

# Location distribution
print(table(analysis_data$location))

# Technical Metrics
# Device distribution
device_dist <- table(analysis_data$device_type)
print(device_dist)
print(prop.table(device_dist) * 100)

# Time of day distribution
print(table(analysis_data$time_of_day))

# Content Metrics Summary
# Headline length
print(summary(analysis_data$headline_length))

# Readability scores
print(summary(analysis_data$body_readability_score))

#===============================================
# Feature Preprocessing
#===============================================

# Create recipe for preprocessing
data_recipe <- recipe(CTR ~ ., data = analysis_data) %>%
  step_impute_mode(all_nominal_predictors()) %>%   # Impute missing categorical values with mode
  step_dummy(all_nominal_predictors()) %>%         # Create dummy variables
  step_impute_bag(all_numeric_predictors()) %>%    # Impute missing numeric values with mean
  step_center(all_numeric_predictors()) %>%        # Center numeric values
  step_normalize(all_numeric_predictors()) %>%     # Normalize numeric values
  prep()

# Apply preprocessing to both datasets
analysis_data_transformed <- bake(data_recipe, new_data = analysis_data)

# Visualize transformed numeric variables
analysis_data_plot <- analysis_data_transformed %>% 
  select(-CTR, -id) %>%
  select_if(is.numeric) %>%
  pivot_longer(everything(), names_to = 'predictor', values_to = 'values') %>%
  ggplot(aes(x = values)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  facet_wrap(predictor ~ ., scales = 'free') +
  labs(title = "Distribution of Transformed Numeric Predictors") +
  theme_bw()

print(analysis_data_plot)

#===============================================
# Train-Test Split
#===============================================

# Set seed
set.seed(1031)

# Split data into training (80%) and test (20%) sets
split = createDataPartition(y = analysis_data_transformed$CTR, p = 0.8, list = F)
train = analysis_data_transformed[split,]
test = analysis_data_transformed[-split,]

#===============================================
# Final Data Cleaning
#===============================================

# Clean column names for consistency
train <- train %>% clean_names()
test <- test %>% clean_names()

# Verify column names
names(train)
names(test)

# Standardize target variable name
train <- train %>% rename(CTR = ctr)
test <- test %>% rename(CTR = ctr)

#===============================================
# Hyperparameter Grid Setup
#===============================================

# Create simplified grid with focused parameter ranges
xgb_grid <- expand.grid(
  eta = c(0.05, 0.1),              # Learning rate - moderate values for balance
  max_depth = c(4, 6),             # Tree depth - medium range for complexity
  min_child_weight = c(1, 3),      # Minimum child weight - favoring detail
  subsample = 0.8,                 # Fixed subsampling rate
  colsample_bytree = 0.8,          # Fixed column sampling rate
  gamma = 0                        # No minimum loss reduction required
)

#===============================================
# Data Treatment Setup
#===============================================

# Create treatment plan for variables
trt = designTreatmentsZ(dframe = train,
                        varlist = names(train)[1:10])

# Select only clean numeric and level-encoded variables
newvars = trt$scoreFrame[trt$scoreFrame$code %in% c('clean','lev'),'varName']

# Apply treatments to create model-ready datasets
train_input = prepare(treatmentplan = trt, 
                      dframe = train,
                      varRestriction = newvars)

test_input = prepare(treatmentplan = trt, 
                     dframe = test,
                     varRestriction = newvars)

#===============================================
# Grid Search with Treated Data
#===============================================

# Initialize results storage
results <- data.frame()

# Perform grid search
for(i in 1:nrow(xgb_grid)) {
  # Evaluate current parameter set
  cv <- xgb.cv(
    data = as.matrix(train_input),
    label = train$CTR,
    params = list(
      eta = xgb_grid$eta[i],
      max_depth = xgb_grid$max_depth[i],
      min_child_weight = xgb_grid$min_child_weight[i],
      subsample = xgb_grid$subsample[i],
      colsample_bytree = xgb_grid$colsample_bytree[i],
      gamma = xgb_grid$gamma[i],
      objective = "reg:squarederror"
    ),
    nrounds = 500,
    nfold = 5,
    early_stopping_rounds = 30,
    verbose = 0,
    metrics = "rmse"
  )
  
  # Store results
  results <- rbind(results, data.frame(
    xgb_grid[i,],
    test_rmse = min(cv$evaluation_log$test_rmse_mean),
    train_rmse = cv$evaluation_log$train_rmse_mean[which.min(cv$evaluation_log$test_rmse_mean)],
    best_nrounds = which.min(cv$evaluation_log$test_rmse_mean)
  ))
  
  cat("Completed parameter set", i, "of", nrow(xgb_grid), "\n")
}

#===============================================
# Final Model Training with Best Parameters
#===============================================

# Find best parameters
best_idx <- which.min(results$test_rmse)
best_params <- as.list(results[best_idx, names(xgb_grid)])
best_nrounds <- results$best_nrounds[best_idx]

# Print best parameters
print(results[best_idx,])

# Train final model
final_model <- xgboost(
  data = as.matrix(train_input),
  label = train$CTR,
  params = c(
    best_params,
    list(objective = "reg:squarederror")
  ),
  nrounds = best_nrounds,
  verbose = 0
)

#===============================================
# Generate Predictions
#===============================================

# Make predictions
pred_train <- predict(final_model, as.matrix(train_input))
pred_test <- predict(final_model, as.matrix(test_input))

# Calculate RMSE
rmse_train <- sqrt(mean((pred_train - train$CTR)^2))
rmse_test <- sqrt(mean((pred_test - test$CTR)^2))

# Print performance metrics
cat("\nFinal Performance Metrics:\n")
cat("Train RMSE:", round(rmse_train, 6), "\n")
cat("Test RMSE:", round(rmse_test, 6), "\n")

# ====================================
# Generate predictions and save results
# ====================================

# Create prediction matrix using same treatment plan as training
analysis_input = prepare(treatmentplan = trt,
                         dframe = analysis_data_transformed,
                         varRestriction = newvars)

predictions <- predict(final_model, as.matrix(analysis_input))

# Create submission with all predictions
submission <- data.frame(
  id = analysis_data$id,
  CTR = predictions
)

# Save with relative path
write.csv(submission, "output/submission_final.csv", row.names = FALSE)
