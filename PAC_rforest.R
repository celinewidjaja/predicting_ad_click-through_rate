# ====================================
# Setup and initialization
# ====================================

# Import essential packages for data manipulation, modeling and preprocessing
library(dplyr)
library(caret)
library(xgboost)
library(tidyr)

# Set random seed for reproducibility of results
set.seed(1031)

# ====================================
# Data loading and initial inspection
# ====================================

# Import the training (analysis) and test (scoring) datasets
analysis_data = read.csv('/Users/celinewidjaja/Desktop/predicting-clicks/analysis_data.csv', stringsAsFactors = TRUE)
scoring_data = read.csv('/Users/celinewidjaja/Desktop/predicting-clicks/scoring_data.csv', stringsAsFactors = TRUE)

# Inspect structure of both datasets
str(analysis_data)
str(scoring_data)

# ====================================
# Data preprocessing
# ====================================

# Specify which columns should be treated as categorical variables
cols_to_factor <- c("contextual_relevance", "seasonality", "headline_power_words",
                    "headline_question", "headline_numbers")

# Convert specified columns to factors in both datasets
analysis_data[cols_to_factor] <- lapply(analysis_data[cols_to_factor], as.factor)
scoring_data[cols_to_factor] <- lapply(scoring_data[cols_to_factor], as.factor)

# Check for duplicate IDs in analysis dataset
analysis_data %>%
  group_by(id) %>%
  count()%>%
  filter(n==1) # Verify uniqueness of IDs

# ====================================
# Missing data handling
# ====================================

# Examine missing values in both datasets
skim(analysis_data)
skim(scoring_data)

# Helper function to calculate proportions of categorical values
# This will be used for imputation based on observed distributions
make_props <- function(data, cols) {
  props_list <- list()
  for(col in cols) {
    props_list[[col]] <- prop.table(table(data[[col]], useNA = "no"))
  }
  return(props_list)
}

# Define categorical columns for proportion calculation
cat_cols <- c("contextual_relevance", "age_group", "gender", "location", 
              "headline_power_words", "headline_question", "headline_numbers")

# Calculate proportions for categorical variables
props <- make_props(analysis_data, cat_cols)

# Helper function to impute missing categorical values
# Uses proportional sampling based on observed distributions
impute_cats <- function(data, cols, props) {
  for(col in cols) {
    data[[col]][is.na(data[[col]])] <- sample(
      names(props[[col]]),  # Sample from unique values
      sum(is.na(data[[col]])), # Number of NAs to impute
      replace = TRUE,
      prob = props[[col]] # Use calculated proportions as sampling weights
    )
  }
  return(data)
}

# Apply categorical imputation to both datasets
analysis_data <- impute_cats(analysis_data, cat_cols, props)
scoring_data <- impute_cats(scoring_data, cat_cols, props)

# Identify numeric columns for mean imputation
num_cols <- names(analysis_data)[sapply(analysis_data, is.numeric)]

# Impute missing numeric values with means from analysis data
for (col in num_cols) {
  if (col %in% names(analysis_data) && col %in% names(scoring_data)) {
    col_mean <- mean(analysis_data[[col]], na.rm = TRUE)
    analysis_data[[col]][is.na(analysis_data[[col]])] <- col_mean
    scoring_data[[col]][is.na(scoring_data[[col]])] <- col_mean
  }
}

# ====================================
# Data validation and range checking
# ====================================

# Function to analyze correlations with target and handle out-of-range values
analyze_numeric <- function(analysis_df, scoring_df, column, min_val, max_val, cap) {
  # Check correlation with target variable (CTR)
  if(column != "CTR") {
    orig_cor <- cor(analysis_df[[column]], analysis_df$CTR, use = "complete.obs")
    cat(sprintf("\nOriginal correlation with CTR for %s: %.3f\n", column, orig_cor))
  }
  
  # Count values outside valid range
  out_range_analysis <- sum(analysis_df[[column]] < min_val | analysis_df[[column]] > max_val, na.rm = TRUE)
  out_range_scoring <- sum(scoring_df[[column]] < min_val | scoring_df[[column]] > max_val, na.rm = TRUE)
  cat(sprintf("\n## %s:\n", column))
  cat(sprintf("Analysis: %d out of range values\n", out_range_analysis))
  cat(sprintf("Scoring: %d out of range values\n", out_range_scoring))
  
  # Cap values at min/max if specified
  if(cap) {
    analysis_df[[column]] <- ifelse(analysis_df[[column]] < min_val, min_val,
                                    ifelse(analysis_df[[column]] > max_val, max_val, 
                                           analysis_df[[column]]))
    if(column != "CTR") {
      scoring_df[[column]] <- ifelse(scoring_df[[column]] < min_val, min_val,
                                     ifelse(scoring_df[[column]] > max_val, max_val, 
                                            scoring_df[[column]]))
      
      new_cor <- cor(analysis_df[[column]], analysis_df$CTR, use = "complete.obs")
      cat(sprintf("Correlation after capping: %.3f\n", new_cor))
    }
    cat(sprintf("Values capped for %s\n", column))
  }
  
  return(list(analysis_df = analysis_df, scoring_df = scoring_df))
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

# Apply range validation and capping to numeric variables
for(var in numeric_vars) {
  results <- analyze_numeric(analysis_data, scoring_data, var$col, var$min, var$max, var$cap)
  analysis_data <- results$analysis_df
  scoring_data <- results$scoring_df
}

# ====================================
# Data standardization
# ====================================

# Create recipe for standardizing numeric features
data_recipe <- recipe(CTR ~ ., data = analysis_dummy) %>%
  step_center(all_numeric_predictors()) %>%  # Center (subtract mean)
  step_normalize(all_numeric_predictors()) %>%  # Scale (divide by std dev)
  prep()

# Apply standardization to both datasets
analysis_dummy <- bake(data_recipe, new_data = analysis_dummy)
scoring_dummy <- bake(data_recipe, new_data = scoring_dummy)

# ====================================
# Train/test split
# ====================================

# Split data into training (80%) and test (20%) sets
set.seed(1031)
split = createDataPartition(y = analysis_dummy$CTR, p = 0.8, list = F)
train = analysis_dummy[split,]
test = analysis_dummy[-split,]

# ====================================
# Data cleaning
# ====================================

# Clean up column names for consistency
library(janitor)
train <- train %>% clean_names()
test <- test %>% clean_names()
scoring_dummy <- scoring_dummy %>% clean_names()

# Verify column names after cleaning
names(train)
names(test)
names(scoring_dummy)

# Ensure target variable has consistent name
train <- train %>%
  rename(CTR = ctr)

test <- test %>%
  rename(CTR = ctr)

# ====================================
# Model fitting and evaluation
# ====================================

# Fit random forest model with 1000 trees
library(randomForest)
set.seed(1031)
forest3 = randomForest(CTR~., 
                       train, 
                       ntree = 1000)

# Calculate training set RMSE
pred_train = predict(forest3)
rmse_train_forest3 = sqrt(mean((pred_train - train$CTR)^2)); rmse_train_forest3

# Calculate test set RMSE
pred_forest = predict(forest3, newdata= test)
rmse_forest3 = sqrt(mean((pred_forest - test$CTR)^2)); rmse_forest3

# ====================================
# Generate predictions and save results
# ====================================

# Generate predictions for scoring data
pred_scoring = predict(forest3, newdata = scoring_dummy)

# Create submission file with predictions
submission <- scoring_data %>%
  dplyr::select(id) %>%
  mutate(CTR = pred_scoring)

# Save predictions to CSV
write.csv(submission, "/Users/celinewidjaja/Desktop/predicting-clicks/submission_forest_proportion_scaling.csv", row.names = FALSE)
