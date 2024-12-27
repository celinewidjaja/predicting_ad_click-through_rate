# ====================================
# Setup and initialization
# ====================================

# Import essential packages for data manipulation, modeling and preprocessing
library(dplyr)
library(caret)
library(xgboost)
library(tidyr)
library(janitor)
library(randomForest)
library(skimr)

# Set random seed for reproducibility of results
set.seed(1031)

# ====================================
# Data loading and initial inspection
# ====================================

# Import the dataset
analysis_data = read.csv('data/analysis_data.csv', stringsAsFactors = TRUE)

# Inspect structure
str(analysis_data)

# ====================================
# Data preprocessing
# ====================================

# Specify which columns should be treated as categorical variables
cols_to_factor <- c("contextual_relevance", "seasonality", "headline_power_words",
                    "headline_question", "headline_numbers")

# Convert specified columns to factors
analysis_data[cols_to_factor] <- lapply(analysis_data[cols_to_factor], as.factor)

# Check for duplicate IDs
analysis_data %>%
  group_by(id) %>%
  count()%>%
  filter(n==1) # Verify uniqueness of IDs

# ====================================
# Missing data handling
# ====================================

# Examine missing values
skim(analysis_data)

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

# Apply categorical imputation
analysis_data <- impute_cats(analysis_data, cat_cols, props)

# Identify numeric columns for mean imputation
num_cols <- names(analysis_data)[sapply(analysis_data, is.numeric)]

# Impute missing numeric values with means
for (col in num_cols) {
  if (col %in% names(analysis_data)) {
    col_mean <- mean(analysis_data[[col]], na.rm = TRUE)
    analysis_data[[col]][is.na(analysis_data[[col]])] <- col_mean
  }
}

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

# ====================================
# Data standardization
# ====================================

# Create recipe for standardizing numeric features
data_recipe <- recipe(CTR ~ ., data = analysis_data) %>%
  step_center(all_numeric_predictors()) %>%  # Center (subtract mean)
  step_normalize(all_numeric_predictors()) %>%  # Scale (divide by std dev)
  prep()

# Apply standardization
analysis_data <- bake(data_recipe, new_data = analysis_data)

# ====================================
# Train/test split
# ====================================

# Set seed
set.seed(1031)

# Split data into training (80%) and test (20%) sets
split = createDataPartition(y = analysis_data$CTR, p = 0.8, list = F)
train = analysis_data[split,]
test = analysis_data[-split,]

# ====================================
# Data cleaning
# ====================================

# Clean up column names for consistency
train <- train %>% clean_names()
test <- test %>% clean_names()

# Verify column names after cleaning
names(train)
names(test)

# Ensure target variable has consistent name
train <- train %>%
  rename(CTR = ctr)

test <- test %>%
  rename(CTR = ctr)

# ====================================
# Model fitting and evaluation
# ====================================

# Set seed
set.seed(1031)

# Fit random forest model with 1000 trees
forest = randomForest(CTR~., 
                       train, 
                       ntree = 1000)

# Calculate training set RMSE
pred_train = predict(forest)
rmse_train_forest3 = sqrt(mean((pred_train - train$CTR)^2)); rmse_train_forest3

# Calculate test set RMSE
pred_forest = predict(forest, newdata= test)
rmse_forest3 = sqrt(mean((pred_forest - test$CTR)^2)); rmse_forest3

# ====================================
# Generate predictions and save results
# ====================================

# Generate predictions for scoring data
predictions = predict(forest, newdata = analysis_data)

# Create submission with all predictions
submission <- data.frame(
  id = analysis_data$id,
  CTR = predictions
)

# Save with relative path
write.csv(submission, "output/submission_forest.csv", row.names = FALSE)
