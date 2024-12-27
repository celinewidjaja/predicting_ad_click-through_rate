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

# Load dataset
analysis_data = read.csv('data/analysis_data.csv', stringsAsFactors = TRUE)

# Examine data structure
str(analysis_data)

# ====================================
# Data preprocessing
# ====================================

# Define and convert categorical columns to factors
cols_to_factor <- c("contextual_relevance", "seasonality", "headline_power_words",
                    "headline_question", "headline_numbers")

analysis_data[cols_to_factor] <- lapply(analysis_data[cols_to_factor], as.factor)

# Check for duplicate IDs
analysis_data %>%
  group_by(id) %>%
  count()%>%
  filter(n==1) # Confirms no duplicates - all 4,000 rows have unique id

# ====================================
# Missing data analysis
# ====================================

# Evaluate missing data
skim(analysis_data)

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

# Apply categorical imputation
analysis_data <- impute_cats(analysis_data, cat_cols, props)

# Impute missing numeric values
num_cols <- names(analysis_data)[sapply(analysis_data, is.numeric)]

# Perform mean imputation for numeric columns using means
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
# Feature engineering
# ====================================

# Create dummy variables for categorical features
dummies <- dummyVars(" ~ .", data = analysis_data %>% select(-CTR))
analysis_dummy <- as.data.frame(predict(dummies, newdata = analysis_data %>% select(-CTR)))

# Add target variable back to analysis data
analysis_dummy$CTR <- analysis_data$CTR

# ====================================
# Train/test split
# ====================================

# Set seed
set.seed(1031)

# Split dataset
split = createDataPartition(y = analysis_dummy$CTR, p = 0.8, list = F)
train = analysis_dummy[split,]
test = analysis_dummy[-split,]

# ====================================
# Model fitting - linear regression
# ====================================

# Fit linear regression model with all features
linear_model = lm(CTR~., data = train)
summary(linear_model)

# ====================================
# Model evaluation
# ====================================

# Calculate training metrics
train_pred = predict(linear_model, newdata = train)
sse10 = sum((train_pred - train$CTR)^2)
sst10 = sum((mean(train$CTR) - train$CTR)^2)
model10_r2 = 1 - sse10/sst10; model10_r2
rmse10 = sqrt(mean((train_pred - train$CTR)^2)); rmse10

# Calculate test metrics
test_pred = predict(linear_model, newdata = test)
rmse10_test = sqrt(mean((test_pred - test$CTR)^2)); rmse10_test

# ====================================
# Generate predictions and save results
# ====================================

# Generate predictions for full dataset
predictions <- predict(linear_model, analysis_dummy)

# Create submission with all predictions
submission <- data.frame(
  id = analysis_data$id,
  CTR = predictions
)

# Save with relative path
write.csv(submission, "output/prediction_linear.csv", row.names = FALSE)
