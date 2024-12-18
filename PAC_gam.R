#===============================================
# Initial Setup and Library Loading
#===============================================
# Load necessary libraries for data manipulation, visualization, and modeling
library(skimr)        # For data summaries
library(dplyr)        # For data manipulation
library(readr)        # For reading CSV files
library(ggplot2)      # For creating plots
library(lattice)      # For advanced visualization
library(caret)        # For train-test splits
library(recipes)      # For data preprocessing
library(vtreat)       # For data preparation
library(xgboost)      # For gradient boosting models
library(Matrix)       # For sparse matrices
library(tidyr)        # For data reshaping
library(tidyverse)    # For tidy data manipulation
library(forcats)      # For factor variable manipulation
library(mgcv)         # For GAM models
library(gridExtra)    # For arranging multiple plots in a grid

# Set random seed for reproducibility
set.seed(1031)

#===============================================
# Data Loading and Initial Inspection
#===============================================
# Load training (analysis) and scoring datasets
analysis_data <- read.csv('/Users/celinewidjaja/Desktop/predicting-clicks/analysis_data.csv', stringsAsFactors = TRUE)
scoring_data <- read.csv('/Users/celinewidjaja/Desktop/predicting-clicks/scoring_data.csv', stringsAsFactors = TRUE)

# Inspect the structure of the datasets
str(analysis_data)
str(scoring_data)

#===============================================
# Data Type Conversion
#===============================================
# Define columns to convert to factors
cols_to_factor <- c("contextual_relevance", "seasonality", "headline_power_words",
                    "headline_question", "headline_numbers")

# Convert specified columns to factors in both datasets
analysis_data[cols_to_factor] <- lapply(analysis_data[cols_to_factor], as.factor)
scoring_data[cols_to_factor] <- lapply(scoring_data[cols_to_factor], as.factor)

#===============================================
# Data Quality Checks
#===============================================
# Check for duplicate IDs in the analysis dataset
analysis_data %>%
  group_by(id) %>%
  count() %>%
  filter(n > 1) # Ensures IDs are unique; no duplicates expected

# Evaluate missing data using skimr
skim(analysis_data)
skim(scoring_data)

#===============================================
# Function Definitions
#===============================================
# Function to analyze correlation impact
analyze_numeric <- function(analysis_df, scoring_df, column, min_val, max_val, cap) {
  # Check original correlation with CTR
  if(column != "CTR") {
    orig_cor <- cor(analysis_df[[column]], analysis_df$CTR, use = "complete.obs")
    cat(sprintf("\nOriginal correlation with CTR for %s: %.3f\n", column, orig_cor))
  }
  
  # Print range violations
  out_range_analysis <- sum(analysis_df[[column]] < min_val | analysis_df[[column]] > max_val, na.rm = TRUE)
  out_range_scoring <- sum(scoring_df[[column]] < min_val | scoring_df[[column]] > max_val, na.rm = TRUE)
  cat(sprintf("\n## %s:\n", column))
  cat(sprintf("Analysis: %d out of range values\n", out_range_analysis))
  cat(sprintf("Scoring: %d out of range values\n", out_range_scoring))
  
  # Apply capping if specified
  if(cap) {
    analysis_df[[column]] <- ifelse(analysis_df[[column]] < min_val, min_val,
                                    ifelse(analysis_df[[column]] > max_val, max_val, 
                                           analysis_df[[column]]))
    if(column != "CTR") {
      scoring_df[[column]] <- ifelse(scoring_df[[column]] < min_val, min_val,
                                     ifelse(scoring_df[[column]] > max_val, max_val, 
                                            scoring_df[[column]]))
      
      # Check correlation after capping
      new_cor <- cor(analysis_df[[column]], analysis_df$CTR, use = "complete.obs")
      cat(sprintf("Correlation after capping: %.3f\n", new_cor))
    }
    cat(sprintf("Values capped for %s\n", column))
  }
  
  return(list(analysis_df = analysis_df, scoring_df = scoring_df))
}

#===============================================
# Variable Processing
#===============================================
# Define numeric variables and their valid ranges
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

categorical_vars <- c("contextual_relevance", "age_group", "gender", "location",
                      "headline_power_words", "headline_question", "headline_numbers",
                      "seasonality")

# Process variables
for(var in numeric_vars) {
  results <- analyze_numeric(analysis_data, scoring_data, var$col, var$min, var$max, var$cap)
  analysis_data <- results$analysis_df
  scoring_data <- results$scoring_df
}

#Check remaining variables
zero_one_vars <- c("body_keyword_density", "body_readability_score")
for(var in zero_one_vars) {
  analysis_range <- any(analysis_data[[var]] > 1 | analysis_data[[var]] < 0, na.rm = TRUE)
  scoring_range <- any(scoring_data[[var]] > 1 | scoring_data[[var]] < 0, na.rm = TRUE)
  
  cat(sprintf("\n## %s range check:\n", var))
  cat("Analysis data out of range:", analysis_range, "\n")
  cat("Scoring data out of range:", scoring_range, "\n")
}

#===============================================
# Harmonize Factor Levels Between Datasets
#===============================================
## Harmonize factor levels
categorical_cols <- names(analysis_data)[sapply(analysis_data, is.factor)]

for (col in categorical_cols) {
  if (col %in% names(scoring_data)) {
    # Convert to character
    analysis_data[[col]] <- as.character(analysis_data[[col]])
    scoring_data[[col]] <- as.character(scoring_data[[col]])
    
    # Get all unique values
    all_levels <- unique(c(analysis_data[[col]], scoring_data[[col]]))
    all_levels <- all_levels[!is.na(all_levels)]
    
    # Convert back to factor
    analysis_data[[col]] <- factor(analysis_data[[col]], levels = all_levels)
    scoring_data[[col]] <- factor(scoring_data[[col]], levels = all_levels)
  }
}

#===============================================
# Exploratory Data Analysis
#===============================================
# Plot numeric variables
numeric_plot <- analysis_data %>% 
  select(-CTR, -id) %>%
  select_if(is.numeric) %>%
  pivot_longer(everything(), names_to = 'predictor', values_to = 'values') %>%
  ggplot(aes(x = values)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  facet_wrap(predictor ~ ., scales = 'free') +
  labs(title = "Distribution of Numeric Predictors") +
  theme_bw()

# Plot categorical variables
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

# Print both plots
print(numeric_plot)
print(categorical_plot)

#===============================================
# Feature Engineering with Recipes
#===============================================
# Create preprocessing recipe
data_recipe <- recipe(CTR ~ ., data = analysis_data) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  prep()

# Apply preprocessing to datasets
analysis_data_transformed <- bake(data_recipe, new_data = analysis_data)
scoring_data_transformed <- bake(data_recipe, new_data = scoring_data)

#===============================================
# Train-Test Split
#===============================================
# Split the transformed data into training and testing sets
split <- createDataPartition(y = analysis_data_transformed$CTR, p = 0.8, list = FALSE)
train <- analysis_data_transformed[split, ]
test <- analysis_data_transformed[-split, ]

#===============================================
# GAM Model Training and Evaluation
#===============================================
# Fit a GAM model to the training data
gam1 <- gam(
  CTR ~ 
    s(targeting_score) + 
    s(visual_appeal) + 
    s(cta_strength) + 
    s(headline_length) + 
    ad_frequency + 
    market_saturation + 
    body_keyword_density + 
    body_readability_score + 
    brand_familiarity + 
    contextual_relevance_X1 + 
    ad_format_Image + 
    ad_format_Video,
  method = "REML",
  data = train
)

# Summarize the model
summary(gam1)

#===============================================
# Evaluate Model Performance
#===============================================
# Predict on train and test sets
pred_train <- predict(gam1, newdata = train)
rmse_train <- sqrt(mean((pred_train - train$CTR)^2))
cat("Train RMSE:", rmse_train, "\n")

pred_test <- predict(gam1, newdata = test)
rmse_test <- sqrt(mean((pred_test - test$CTR)^2))
cat("Test RMSE:", rmse_test, "\n")

#===============================================
# Generate Predictions for Scoring Data
#===============================================
# Predict CTR for scoring dataset
pred_scoring <- predict(gam1, newdata = scoring_data_transformed)

# Prepare submission dataframe
submission <- scoring_data %>%
  select(id) %>%
  mutate(CTR = pred_scoring)

# Save submission file
write.csv(submission, "/Users/celinewidjaja/Desktop/predicting-clicks/submission_gam1.csv", row.names = FALSE)
