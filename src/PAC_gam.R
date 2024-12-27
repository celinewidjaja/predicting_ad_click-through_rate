#===============================================
# Setup and initialization
#===============================================

# Load necessary libraries for data manipulation, visualization, and modeling
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
library(mgcv)
library(gridExtra)

# Set random seed for reproducibility
set.seed(1031)

#===============================================
# Data Loading and Initial Inspection
#===============================================

# Load dataset
analysis_data = read.csv('data/analysis_data.csv', stringsAsFactors = TRUE)

# Inspect the structure
str(analysis_data)

#===============================================
# Data Type Conversion
#===============================================

# Define columns to convert to factors
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
  filter(n > 1) # Ensures IDs are unique; no duplicates expected

# Evaluate missing data using skimr
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

# Apply preprocessing
analysis_data_transformed <- bake(data_recipe, new_data = analysis_data)

#===============================================
# Train-Test Split
#===============================================

# Set seed
set.seed(1031)

# Split data into training (80%) and test (20%) sets
split <- createDataPartition(y = analysis_data_transformed$CTR, p = 0.8, list = FALSE)
train <- analysis_data_transformed[split, ]
test <- analysis_data_transformed[-split, ]

#===============================================
# GAM Model Training and Evaluation
#===============================================

# Fit a GAM model to the training data
names(train)
gam <- gam(
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
    ad_format_Text + 
    ad_format_Video,
  method = "REML",
  data = train
)

# Summarize the model
summary(gam)

#===============================================
# Evaluate Model Performance
#===============================================

# Predict on train and test sets
pred_train <- predict(gam, newdata = train)
rmse_train <- sqrt(mean((pred_train - train$CTR)^2))
cat("Train RMSE:", rmse_train, "\n")

pred_test <- predict(gam, newdata = test)
rmse_test <- sqrt(mean((pred_test - test$CTR)^2))
cat("Test RMSE:", rmse_test, "\n")

# ====================================
# Generate predictions and save results
# ====================================

# Generate predictions for full dataset
predictions <- predict(gam, analysis_data_transformed)

# Create submission with all predictions
submission <- data.frame(
  id = analysis_data$id,
  CTR = predictions
)

# Save with relative path
write.csv(submission, "output/prediction_gam.csv", row.names = FALSE)
