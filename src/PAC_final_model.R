#===============================================
# Initial Setup and Library Loading
#===============================================
# Load libraries for data manipulation, visualization, and modeling
library(skimr)        # For data summaries
library(dplyr)        # For data manipulation
library(readr)        # For reading data
library(ggplot2)      # For visualization
library(lattice)      # For visualization
library(caret)        # For machine learning
library(recipes)      # For feature engineering
library(vtreat)       # For data treatment
library(xgboost)      # For gradient boosting
library(Matrix)       # For sparse matrices
library(tidyr)        # For data tidying
library(tidyverse)    # For data manipulation
library(forcats)      # For factor manipulation
library(mgcv)         # For GAM models
library(gridExtra)    # For arranging plots
library(janitor)      # For cleaning column names

# Set random seed for reproducibility
set.seed(1031)

#===============================================
# Data Loading and Initial Inspection
#===============================================
# Load analysis and scoring datasets
analysis_data = read.csv('/Users/celinewidjaja/Desktop/predicting-clicks/analysis_data.csv', stringsAsFactors = TRUE)
scoring_data = read.csv('/Users/celinewidjaja/Desktop/predicting-clicks/scoring_data.csv', stringsAsFactors = TRUE)

# Inspect data structure
str(analysis_data)
str(scoring_data)

#===============================================
# Data Type Conversion
#===============================================
# Define columns that should be treated as factors
cols_to_factor <- c("contextual_relevance", "seasonality", "headline_power_words",
                    "headline_question", "headline_numbers")

# Convert specified columns to factors in both datasets
analysis_data[cols_to_factor] <- lapply(analysis_data[cols_to_factor], as.factor)
scoring_data[cols_to_factor] <- lapply(scoring_data[cols_to_factor], as.factor)

#===============================================
# Data Quality Checks
#===============================================
# Check for duplicate IDs
analysis_data %>%
  group_by(id) %>%
  count() %>%
  filter(n==1) # Confirms no duplicates - all 4,000 rows have unique IDs

# Evaluate missing data in both datasets
skim(analysis_data)
skim(scoring_data)

#===============================================
# Outlier Analysis
#===============================================
# Create boxplot to identify CTR outliers
ggplot(data=analysis_data_transformed, aes(x='', y=CTR)) +
  geom_boxplot(outlier.color='red', outlier.alpha=0.5, fill='cadetblue') +
  geom_text(aes(x='', y=median(analysis_data_transformed$CTR), 
                label=median(analysis_data_transformed$CTR)), size=3, hjust=11) +
  xlab(label = '') +
  theme_bw() +
  labs(title = 'Identify Outliers', x = '')

#===============================================
# Define Data Validation Function
#===============================================
# Function to analyze and handle out-of-range values
analyze_numeric <- function(analysis_df, scoring_df, column, min_val, max_val, cap) {
  # Calculate correlation with CTR (if not CTR column)
  if(column != "CTR") {
    orig_cor <- cor(analysis_df[[column]], analysis_df$CTR, use = "complete.obs")
    cat(sprintf("\nOriginal correlation with CTR for %s: %.3f\n", column, orig_cor))
  }
  
  # Check for out-of-range values
  out_range_analysis <- sum(analysis_df[[column]] < min_val | analysis_df[[column]] > max_val, na.rm = TRUE)
  out_range_scoring <- sum(scoring_df[[column]] < min_val | scoring_df[[column]] > max_val, na.rm = TRUE)
  cat(sprintf("\n## %s:\n", column))
  cat(sprintf("Analysis: %d out of range values\n", out_range_analysis))
  cat(sprintf("Scoring: %d out of range values\n", out_range_scoring))
  
  # Cap values if specified
  if(cap) {
    analysis_df[[column]] <- ifelse(analysis_df[[column]] < min_val, min_val,
                                    ifelse(analysis_df[[column]] > max_val, max_val, 
                                           analysis_df[[column]]))
    if(column != "CTR") {
      scoring_df[[column]] <- ifelse(scoring_df[[column]] < min_val, min_val,
                                     ifelse(scoring_df[[column]] > max_val, max_val, 
                                            scoring_df[[column]]))
      
      # Recalculate correlation after capping
      new_cor <- cor(analysis_df[[column]], analysis_df$CTR, use = "complete.obs")
      cat(sprintf("Correlation after capping: %.3f\n", new_cor))
    }
    cat(sprintf("Values capped for %s\n", column))
  }
  
  return(list(analysis_df = analysis_df, scoring_df = scoring_df))
}

#===============================================
# Define Variables for Validation
#===============================================
# List of numeric variables with their valid ranges and capping rules
numeric_vars <- list(
  list(col = "targeting_score", min = 1, max = 10, cap = FALSE),
  list(col = "visual_appeal", min = 1, max = 10, cap = FALSE), 
  list(col = "cta_strength", min = 1, max = 10, cap = FALSE),
  list(col = "brand_familiarity", min = 1, max = 10, cap = TRUE),
  list(col = "market_saturation", min = 1, max = 10, cap = TRUE),
  list(col = "body_keyword_density", min = 0, max = 1, cap = FALSE),
  list(col = "body_readability_score", min = 1, max = 100, cap = FALSE),
  list(col = "CTR", min = 0, max = 1, cap = TRUE)  # CTR always capped between 0-1
)

#===============================================
# Data Validation and Cleaning
#===============================================
# Process all numeric variables
for(var in numeric_vars) {
  results <- analyze_numeric(analysis_data, scoring_data, var$col, var$min, var$max, var$cap)
  analysis_data <- results$analysis_df
  scoring_data <- results$scoring_df
}

# Check variables that should be between 0 and 1
zero_one_vars <- c("body_keyword_density", "body_readability_score")
for(var in zero_one_vars) {
  analysis_range <- any(analysis_data[[var]] > 1 | analysis_data[[var]] < 0, na.rm = TRUE)
  scoring_range <- any(scoring_data[[var]] > 1 | scoring_data[[var]] < 0, na.rm = TRUE)
  
  cat(sprintf("\n## %s range check:\n", var))
  cat("Analysis data out of range:", analysis_range, "\n")
  cat("Scoring data out of range:", scoring_range, "\n")
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

# Create same features in scoring data
scoring_data <- scoring_data %>%
  mutate(
    content_score = (visual_appeal + targeting_score + cta_strength) / 3,
    peak_hour = ifelse(time_of_day %in% c("Morning", "Afternoon"), 1, 0)
  )

#===============================================
# Factor Level Harmonization
#===============================================
# Ensure factor levels are consistent between datasets
categorical_cols <- names(analysis_data)[sapply(analysis_data, is.factor)]

for (col in categorical_cols) {
  if (col %in% names(scoring_data)) {
    # Convert to character temporarily
    analysis_data[[col]] <- as.character(analysis_data[[col]])
    scoring_data[[col]] <- as.character(scoring_data[[col]])
    
    # Get all unique levels
    all_levels <- unique(c(analysis_data[[col]], scoring_data[[col]]))
    all_levels <- all_levels[!is.na(all_levels)]
    
    # Convert back to factor with harmonized levels
    analysis_data[[col]] <- factor(analysis_data[[col]], levels = all_levels)
    scoring_data[[col]] <- factor(scoring_data[[col]], levels = all_levels)
  }
}

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
scoring_data_transformed <- bake(data_recipe, new_data = scoring_data)

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
# Create 80-20 split for training and testing
set.seed(1031)
split = createDataPartition(y = analysis_data_transformed$CTR, p = 0.8, list = F)
train = analysis_data_transformed[split,]
test = analysis_data_transformed[-split,]

#===============================================
# Final Data Cleaning
#===============================================
# Clean column names for consistency
train <- train %>% clean_names()
test <- test %>% clean_names()
scoring_dummy <- scoring_dummy %>% clean_names()

# Verify column names
names(train)
names(test)
names(scoring_dummy)

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

scoring_input = prepare(treatmentplan = trt, 
                        dframe = scoring_data_transformed,
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
pred_scoring <- predict(final_model, as.matrix(scoring_input))

# Calculate RMSE
rmse_train <- sqrt(mean((pred_train - train$CTR)^2))
rmse_test <- sqrt(mean((pred_test - test$CTR)^2))

# Print performance metrics
cat("\nFinal Performance Metrics:\n")
cat("Train RMSE:", round(rmse_train, 6), "\n")
cat("Test RMSE:", round(rmse_test, 6), "\n")

#===============================================
# Create Submission
#===============================================
# Create submission dataframe
submission <- data.frame(
  id = scoring_data$id,
  CTR = pred_scoring
)

# Save predictions
write.csv(submission, 
          "submission22.csv", 
          row.names = FALSE)
