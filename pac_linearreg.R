# ====================================
# Setup and initialization
# ====================================

# Load essential libraries for data manipulation and modeling
library(dplyr)
library(caret)
library(xgboost)

# Set random seed for reproducibility
set.seed(1031)

# ====================================
# Data loading and initial inspection
# ====================================

# Load analysis and scoring datasets
analysis_data = read.csv('/Users/celinewidjaja/Desktop/predicting-clicks/analysis_data.csv', stringsAsFactors = TRUE)
scoring_data = read.csv('/Users/celinewidjaja/Desktop/predicting-clicks/scoring_data.csv', stringsAsFactors = TRUE)

# Examine data structure
str(analysis_data)
str(scoring_data)

# ====================================
# Data preprocessing
# ====================================

# Define and convert categorical columns to factors
cols_to_factor <- c("contextual_relevance", "seasonality", "headline_power_words",
                    "headline_question", "headline_numbers")

analysis_data[cols_to_factor] <- lapply(analysis_data[cols_to_factor], as.factor)
scoring_data[cols_to_factor] <- lapply(scoring_data[cols_to_factor], as.factor)

# Check for duplicate IDs in analysis data
analysis_data %>%
  group_by(id) %>%
  count()%>%
  filter(n==1) # Confirms no duplicates - all 4,000 rows have unique id

# ====================================
# Missing data analysis
# ====================================

# Evaluate missing data patterns in both datasets
skim(analysis_data)
skim(scoring_data)

# Function to calculate proportions for categorical variables
make_props <- function(data, cols) {
  props_list <- list()
  for(col in cols) {
    props_list[[col]] <- prop.table(table(data[[col]], useNA = "no"))
  }
  return(props_list)
}

# Define categorical columns and calculate their proportions
cat_cols <- c("contextual_relevance", "age_group", "gender", "location", 
              "headline_power_words", "headline_question", "headline_numbers")

props <- make_props(analysis_data, cat_cols)

# ====================================
# Missing data imputation
# ====================================

# Function to impute missing categorical values based on proportions
impute_cats <- function(data, cols, props) {
  for(col in cols) {
    data[[col]][is.na(data[[col]])] <- sample(
      names(props[[col]]),  # Changed from props[,col] to props[[col]]
      sum(is.na(data[[col]])),
      replace = TRUE,
      prob = props[[col]]
    )
  }
  return(data)
}

# Apply categorical imputation to both datasets
analysis_data <- impute_cats(analysis_data, cat_cols, props)
scoring_data <- impute_cats(scoring_data, cat_cols, props)

# Impute missing numeric values
num_cols <- names(analysis_data)[sapply(analysis_data, is.numeric)]

# Perform mean imputation for numeric columns using analysis data means
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

# Function to analyze correlations and handle out-of-range values
analyze_numeric <- function(analysis_df, scoring_df, column, min_val, max_val, cap) {
  # Check correlation with CTR
  if(column != "CTR") {
    orig_cor <- cor(analysis_df[[column]], analysis_df$CTR, use = "complete.obs")
    cat(sprintf("\nOriginal correlation with CTR for %s: %.3f\n", column, orig_cor))
  }
  
  # Identify out-of-range values
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
      
      new_cor <- cor(analysis_df[[column]], analysis_df$CTR, use = "complete.obs")
      cat(sprintf("Correlation after capping: %.3f\n", new_cor))
    }
    cat(sprintf("Values capped for %s\n", column))
  }
  
  return(list(analysis_df = analysis_df, scoring_df = scoring_df))
}

# Define numeric variable constraints and capping rules
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

# Process and validate numeric variables
for(var in numeric_vars) {
  results <- analyze_numeric(analysis_data, scoring_data, var$col, var$min, var$max, var$cap)
  analysis_data <- results$analysis_df
  scoring_data <- results$scoring_df
}

# ====================================
# Feature engineering
# ====================================

# Harmonize factor levels between datasets
for (col in cat_cols) {
  analysis_data[[col]] <- factor(analysis_data[[col]])
  scoring_data[[col]] <- factor(scoring_data[[col]], levels = levels(analysis_data[[col]]))
}

# Create dummy variables for categorical features
dummies <- dummyVars(" ~ .", data = analysis_data %>% select(-CTR))
analysis_dummy <- as.data.frame(predict(dummies, newdata = analysis_data %>% select(-CTR)))
scoring_dummy <- as.data.frame(predict(dummies, newdata = scoring_data))

# Add target variable back to analysis data
analysis_dummy$CTR <- analysis_data$CTR

# ====================================
# Train/test split
# ====================================

set.seed(1031)
split = createDataPartition(y = analysis_dummy$CTR, p = 0.8, list = F)
train = analysis_dummy[split,]
test = analysis_dummy[-split,]

# ====================================
# Model fitting - linear regression
# ====================================

# Fit linear regression model with all features
linear_model10 = lm(CTR~., data = train)
summary(linear_model10)

# ====================================
# Model evaluation
# ====================================

# Calculate training metrics
train_pred = predict(linear_model10, newdata = train)
sse10 = sum((train_pred - train$CTR)^2)
sst10 = sum((mean(train$CTR) - train$CTR)^2)
model10_r2 = 1 - sse10/sst10; model10_r2
rmse10 = sqrt(mean((train_pred - train$CTR)^2)); rmse10

# Calculate test metrics
test_pred = predict(linear_model10, newdata = test)
rmse10_test = sqrt(mean((test_pred - test$CTR)^2)); rmse10_test

# ====================================
# Generate predictions and save results
# ====================================

# Generate predictions for scoring data
pred_scoring = predict(linear_model10, newdata = scoring_dummy)

# Create submission file
submission <- scoring_data %>%
  select(id) %>%
  mutate(CTR = pred_scoring)

# Save predictions
write.csv(submission, "/Users/celinewidjaja/Desktop/predicting-clicks/submission_lm10_proportion.csv", row.names = FALSE)