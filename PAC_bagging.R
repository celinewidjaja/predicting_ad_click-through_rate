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

# Set random seed for reproducibility
set.seed(1031)

# ====================================
# Data loading and initial inspection
# ====================================

# Import the training (analysis) and testing (scoring) datasets
analysis_data = read.csv('/Users/celinewidjaja/Desktop/predicting-clicks/analysis_data.csv', stringsAsFactors = TRUE)
scoring_data = read.csv('/Users/celinewidjaja/Desktop/predicting-clicks/scoring_data.csv', stringsAsFactors = TRUE)

# Examine data structure
str(analysis_data)
str(scoring_data)

# ====================================
# Initial data preprocessing
# ====================================

# Define categorical variables that need to be converted to factors
cols_to_factor <- c("contextual_relevance", "seasonality", "headline_power_words",
                    "headline_question", "headline_numbers")

# Convert specified columns to factors in both datasets
analysis_data[cols_to_factor] <- lapply(analysis_data[cols_to_factor], as.factor)
scoring_data[cols_to_factor] <- lapply(scoring_data[cols_to_factor], as.factor)

# Check for duplicate IDs in the analysis dataset
analysis_data %>%
  group_by(id) %>%
  count()%>%
  filter(n==1) # Verifies that all 4,000 rows have unique IDs

# ====================================
# Missing data analysis
# ====================================

# Examine missing values in both datasets
skim(analysis_data)
skim(scoring_data)

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

# Define categorical variables
categorical_vars <- c("contextual_relevance", "age_group", "gender", "location",
                      "headline_power_words", "headline_question", "headline_numbers",
                      "seasonality")

# Apply range validation and capping to numeric variables
for(var in numeric_vars) {
  results <- analyze_numeric(analysis_data, scoring_data, var$col, var$min, var$max, var$cap)
  analysis_data <- results$analysis_df
  scoring_data <- results$scoring_df
}

# ====================================
# Range validation for special variables
# ====================================

# Check variables that should be between 0 and 1
zero_one_vars <- c("body_keyword_density", "body_readability_score")
for(var in zero_one_vars) {
  analysis_range <- any(analysis_data[[var]] > 1 | analysis_data[[var]] < 0, na.rm = TRUE)
  scoring_range <- any(scoring_data[[var]] > 1 | scoring_data[[var]] < 0, na.rm = TRUE)
  
  cat(sprintf("\n## %s range check:\n", var))
  cat("Analysis data out of range:", analysis_range, "\n")
  cat("Scoring data out of range:", scoring_range, "\n")
}

# ====================================
# Factor level harmonization
# ====================================

# Ensure factor levels match between datasets
categorical_cols <- names(analysis_data)[sapply(analysis_data, is.factor)]

for (col in categorical_cols) {
  if (col %in% names(scoring_data)) {
    # Convert to character for level matching
    analysis_data[[col]] <- as.character(analysis_data[[col]])
    scoring_data[[col]] <- as.character(scoring_data[[col]])
    
    # Get unified set of levels
    all_levels <- unique(c(analysis_data[[col]], scoring_data[[col]]))
    all_levels <- all_levels[!is.na(all_levels)]
    
    # Convert back to factor with harmonized levels
    analysis_data[[col]] <- factor(analysis_data[[col]], levels = all_levels)
    scoring_data[[col]] <- factor(scoring_data[[col]], levels = all_levels)
  }
}

# ====================================
# Data visualization
# ====================================

# Create histogram plots for numeric variables
numeric_plot <- analysis_data %>% 
  dplyr::select(-"CTR", -"id") %>%
  select_if(is.numeric) %>%
  pivot_longer(everything(), names_to = 'predictor', values_to = 'values') %>%
  ggplot(aes(x = values)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  facet_wrap(predictor ~ ., scales = 'free') +
  labs(title = "Distribution of Numeric Predictors") +
  theme_bw()

# Create bar plots for categorical variables
categorical_plot <- analysis_data %>% 
  dplyr::select(-"CTR", -"id") %>%
  select_if(is.factor) %>%
  pivot_longer(everything(), names_to = 'predictor', values_to = 'values') %>%
  ggplot(aes(x = values)) +
  geom_bar(fill = "lightgreen", color = "black") +
  facet_wrap(predictor ~ ., scales = 'free') +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Distribution of Categorical Predictors")

# Display distribution plots
print(numeric_plot)
print(categorical_plot)

# ====================================
# Feature engineering and transformation
# ====================================

# Create preprocessing recipe for data transformation
data_recipe <- recipe(CTR ~ ., data = analysis_data) %>%
  step_impute_mode(all_nominal_predictors()) %>%  # Impute missing categorical values with mode
  step_dummy(all_nominal_predictors()) %>%        # Convert categorical to dummy variables
  step_impute_median(all_numeric_predictors()) %>% # Impute missing numeric values with median
  prep()

# Apply preprocessing transformations
analysis_data_transformed <- bake(data_recipe, new_data = analysis_data)
scoring_data_transformed <- bake(data_recipe, new_data = scoring_data)

# ====================================
# Post-transformation visualization
# ====================================

# Plot distributions after transformation
analysis_data_plot <- analysis_data_transformed %>% 
  dplyr::select(-"CTR", -"id") %>%
  select_if(is.numeric) %>%
  pivot_longer(everything(), names_to = 'predictor', values_to = 'values') %>%
  ggplot(aes(x = values)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  facet_wrap(predictor ~ ., scales = 'free') +
  labs(title = "Distribution of Numeric Predictors") +
  theme_bw()

print(analysis_data_plot)

# Create boxplot to examine CTR outliers
ggplot(data=analysis_data_transformed,aes(x='',y=CTR))+
  geom_boxplot(outlier.color='red',outlier.alpha=0.5, fill='cadetblue')+
  geom_text(aes(x='',y=median(analysis_data_transformed$CTR),label=median(analysis_data_transformed$CTR)),size=3,hjust=11)+
  xlab(label = '')+
  theme_bw()+
  labs(title = 'Identify Outliers', x = '')

# ====================================
# Train/test split
# ====================================

# Split data into training (80%) and test (20%) sets
set.seed(1031)
split = createDataPartition(y = analysis_data_transformed$CTR, p = 0.8, list = F)
train = analysis_data_transformed[split,]
test = analysis_data_transformed[-split,]

# ====================================
# Model fitting and evaluation
# ====================================

# Fit bagging model with 1000 trees using ipred package
library(ipred)
set.seed(1031) 
bag1 = bagging(CTR~.,
               data = train, 
               nbagg = 1000)

# Calculate training RMSE
pred = predict(bag1, train)
rmse_train_bag_ipred = sqrt(mean((pred - train$CTR)^2))
rmse_train_bag_ipred

# Calculate test RMSE
pred_test = predict(bag1, test)
rmse_test_bag_ipred = sqrt(mean((pred_test - test$CTR)^2))
rmse_test_bag_ipred

# ====================================
# Generate predictions and save results
# ====================================

# Generate predictions for scoring data
pred_scoring = predict(bag1, newdata = scoring_data_transformed)

# Create submission file with predictions
submission <- scoring_data %>%
  dplyr::select(id) %>%
  mutate(CTR = pred_scoring)

# Save predictions to CSV
write.csv(submission, "/Users/celinewidjaja/Desktop/predicting-clicks/submission_bag.csv", row.names = FALSE)