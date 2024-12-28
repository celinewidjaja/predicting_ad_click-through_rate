# ====================================
# Setup and initialization
# ====================================

# Load required libraries for data manipulation, visualization, and modeling
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
library(leaps)
library(broom)
library(car)
library(MASS)
library(glmnet)

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
# Initial data preprocessing
# ====================================

# Define categorical variables that need to be converted to factors
cols_to_factor <- c("contextual_relevance", "seasonality", "headline_power_words",
                    "headline_question", "headline_numbers")

# Convert specified columns to factors
analysis_data[cols_to_factor] <- lapply(analysis_data[cols_to_factor], as.factor)

# Check for duplicate IDs
analysis_data %>%
  group_by(id) %>%
  count()%>%
  filter(n==1) # Verifies that all 4,000 rows have unique IDs

# ====================================
# Missing data analysis
# ====================================

# Examine missing values
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

# ====================================
# Feature engineering
# ====================================

# Function to create polynomial features based on GAM analysis
create_poly_features <- function(data) {
  data %>%
    # Handle missing values with median imputation
    mutate(
      targeting_score = ifelse(is.na(targeting_score), median(targeting_score, na.rm=TRUE), targeting_score),
      visual_appeal = ifelse(is.na(visual_appeal), median(visual_appeal, na.rm=TRUE), visual_appeal),
      headline_length = ifelse(is.na(headline_length), median(headline_length, na.rm=TRUE), headline_length),
      cta_strength = ifelse(is.na(cta_strength), median(cta_strength, na.rm=TRUE), cta_strength)
    ) %>%
    # Create polynomial features
    mutate(
      targeting_score_poly2 = poly(targeting_score, degree = 3)[,2],
      targeting_score_poly3 = poly(targeting_score, degree = 3)[,3],
      visual_appeal_poly2 = poly(visual_appeal, degree = 3)[,2],
      visual_appeal_poly3 = poly(visual_appeal, degree = 3)[,3],
      headline_length_poly2 = poly(headline_length, degree = 2)[,2],
      cta_strength_poly2 = poly(cta_strength, degree = 2)[,2]
    )
}

# Apply polynomial transformations
analysis_data <- create_poly_features(analysis_data)

# ====================================
# Exploratory data analysis
# ====================================

# Plot distributions of numeric variables
numeric_plot <- analysis_data %>% 
  dplyr::select(-CTR, -id) %>%
  dplyr::select_if(is.numeric) %>%
  pivot_longer(everything(), names_to = 'predictor', values_to = 'values') %>%
  ggplot(aes(x = values)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  facet_wrap(predictor ~ ., scales = 'free') +
  labs(title = "Distribution of Numeric Predictors") +
  theme_bw()

# Plot distributions of categorical variables
categorical_plot <- analysis_data %>% 
  dplyr::select(-CTR, -id) %>%
  dplyr::select_if(is.factor) %>%
  pivot_longer(everything(), names_to = 'predictor', values_to = 'values') %>%
  ggplot(aes(x = values)) +
  geom_bar(fill = "lightgreen", color = "black") +
  facet_wrap(predictor ~ ., scales = 'free') +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Distribution of Categorical Predictors")

print(numeric_plot)
print(categorical_plot)

# ====================================
# Data transformation
# ====================================

# Create and apply preprocessing recipe
data_recipe <- recipe(CTR ~ ., data = analysis_data) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  prep()

analysis_data_transformed <- bake(data_recipe, new_data = analysis_data)

# Visualize transformed numeric variables
analysis_data_plot <- analysis_data_transformed %>% 
  dplyr::select(-CTR, -id) %>%
  dplyr::select_if(is.numeric) %>%
  pivot_longer(everything(), names_to = 'predictor', values_to = 'values') %>%
  ggplot(aes(x = values)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  facet_wrap(predictor ~ ., scales = 'free') +
  labs(title = "Distribution of Numeric Predictors") +
  theme_bw()

print(analysis_data_plot)

# Examine CTR outliers
CTR_plot <- ggplot(data=analysis_data_transformed,aes(x='',y=CTR))+
  geom_boxplot(outlier.color='red',outlier.alpha=0.5, fill='cadetblue')+
  geom_text(aes(x='',y=median(analysis_data_transformed$CTR),label=median(analysis_data_transformed$CTR)),size=3,hjust=11)+
  xlab(label = '')+
  theme_bw()+
  labs(title = 'Identify Outliers', x = '')

# ====================================
# Train/test split
# ====================================

# Set seed
set.seed(1031)

# Split data into training (80%) and test (20%) sets
split = createDataPartition(y = analysis_data_transformed$CTR, p = 0.8, list = F)
train = analysis_data_transformed[split,]
test = analysis_data_transformed[-split,]

# ====================================
# Correlation analysis
# ====================================

# Visualize correlations
library(ggcorrplot)
corplot <- ggcorrplot(cor(train),
           method = 'square',
           type = 'lower',
           show.diag = F,
           colors = c('#e9a3c9', '#f7f7f7', '#a1d76a'))

correlationMatrix = cor(train)
View(round(cor(train), 2)*100)

# ====================================
# Initial model fitting and diagnostics
# ====================================

# Fit initial model
model = lm(CTR~.,train)
summary(model)

# Print tidy summary
summary(model) %>%
  tidy()

# Check multicollinearity
vif(model)

# Visualize VIF values
vif_plot <- data.frame(Predictor = names(vif(model)), VIF = vif(model)) %>%
  ggplot(aes(x=VIF, y = reorder(Predictor, VIF), fill=VIF))+
  geom_col()+
  geom_vline(xintercept=5, color = 'gray', size = 1.5)+
  geom_vline(xintercept = 10, color = 'red', size = 1.5)+
  scale_fill_gradient(low = '#fff7bc', high = '#d95f0e')+
  scale_y_discrete(name = "Predictor")+
  scale_x_continuous(breaks = seq(5,30,5))+
  theme_classic()

# ====================================
# Best subset selection
# ====================================

subsets = regsubsets(CTR~.,data=train, nvmax=15, really.big=T)
summary(subsets)

# Create dataframe of selection criteria
subsets_measures = data.frame(model=1:length(summary(subsets)$cp),
                              cp=summary(subsets)$cp,
                              bic=summary(subsets)$bic, 
                              adjr2=summary(subsets)$adjr2)

# Plot selection criteria
subset_selection_plot <- subsets_measures %>%
  gather(key = type, value=value, 2:4)%>%
  group_by(type)%>%
  mutate(best_value = factor(ifelse(value == min(value) | value== max(value),0,1)))%>%
  ungroup()%>%
  ggplot(aes(x=model,y=value))+
  geom_line(color='gray2')+
  geom_point(aes(color = best_value), size=2.5)+
  scale_x_discrete(limits = seq(1,15,1),name = 'Number of Variables')+
  scale_y_continuous(name = '')+
  guides(color=F)+
  theme_bw()+
  facet_grid(type~.,scales='free_y')

# ====================================
# Forward selection
# ====================================

# Initialize models for selection
start_mod = lm(CTR~1, data=train)
empty_mod = lm(CTR~1, data=train)
full_mod = lm(CTR~., data=train)

# Perform forward selection
forwardStepwise = stats::step(start_mod,
                              scope=list(upper=full_mod, lower=empty_mod),
                              direction='forward',
                              trace=FALSE)

# Visualize AIC progression
stepwise_plot <- forwardStepwise$anova %>% 
  mutate(step_number = as.integer(rownames(forwardStepwise$anova))-1) %>%
  mutate(Step = as.character(Step))%>%
  ggplot(aes(x = reorder(Step,X = step_number), y = AIC))+
  geom_point(color = 'darkgreen', size = 2) + 
  scale_x_discrete(name = 'Variable Entering Model')+
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.9, hjust=0.9))

# ====================================
# Backward selection
# ====================================

# Initialize models
start_mod = lm(CTR~.,data=train)
empty_mod = lm(CTR~1,data=train)
full_mod = lm(CTR~.,data=train)

# Perform backward selection
backwardStepwise = stepAIC(start_mod, 
                           scope=list(upper=full_mod, lower=empty_mod),
                           direction="backward",
                           trace=FALSE)

# View results
summary(backwardStepwise)

# Visualize AIC progression
backward_plot <- backwardStepwise$anova %>% 
  mutate(step_number = as.integer(rownames(backwardStepwise$anova))-1) %>%
  mutate(Step = as.character(Step))%>%
  ggplot(aes(x = reorder(Step,X = step_number), y = AIC))+
  geom_point(color = 'darkgreen', size = 2) + 
  scale_x_discrete(name = 'Variable Dropped')+
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.9, hjust=0.9))

# ====================================
# Stepwise selection
# ====================================

# Initialize models
start_mod = lm(CTR~1,data=train)
empty_mod = lm(CTR~1,data=train)
full_mod = lm(CTR~.,data=train)

# Perform stepwise selection
hybridStepwise = stepAIC(start_mod, 
                         scope=list(upper=full_mod, lower=empty_mod),
                         direction="both",
                         trace=FALSE)

summary(hybridStepwise)

# Visualize AIC progression
hybridstep_plot <- hybridStepwise$anova %>% 
  mutate(step_number = as.integer(rownames(hybridStepwise$anova))-1) %>%
  mutate(Step = as.character(Step))%>%
  ggplot(aes(x = reorder(Step,X = step_number), y = AIC))+
  geom_point(color = 'darkgreen', size = 2) + 
  scale_x_discrete(name = 'Variable Added or Dropped')+
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.9, hjust=0.9))

# ====================================
# Ridge regression
# ====================================

# Prepare model matrix
x = model.matrix(CTR~.-1,data=train)
y = train$CTR

# Fit ridge regression
set.seed(617)
cv_ridge = glmnet(x = x, y = y, alpha = 0)

# Cross-validation for lambda selection
cv_ridge = cv.glmnet(x, y, alpha = 0)
optimal_lambda = cv_ridge$lambda.min

# Extract coefficients at optimal lambda
coef_df = data.frame(
  feature = rownames(coef(cv_ridge, s = optimal_lambda)),
  coefficient = as.numeric(coef(cv_ridge, s = optimal_lambda))
) %>%
  arrange(desc(abs(coefficient))) %>%
  filter(coefficient != 0)

# View important coefficients
head(coef_df, 12)

# Visualize coefficient paths
plot_df <- data.frame(
  lambda = rep(cv_ridge$lambda, each = nrow(coef(cv_ridge))),
  coefficient = as.vector(as.matrix(coef(cv_ridge))),
  feature = rep(rownames(coef(cv_ridge)), length(cv_ridge$lambda))
) %>%
  filter(feature != "(Intercept)")

# Plot ridge paths
ridge_plot <- ggplot(plot_df, aes(x = log(lambda), y = coefficient, color = feature)) +
  geom_line() +
  theme_bw() +
  labs(title = "Ridge Coefficient Paths",
       x = "log(lambda)",
       y = "Coefficients") +
  theme(legend.position = "right")

# ====================================
# Lasso regression
# ====================================

# Set seed
set.seed(617)
  
# Fit lasso model
cv_lasso = cv.glmnet(x = x, 
                     y = y, 
                     alpha = 1,
                     type.measure = 'mse')

# Plot lasso results
plot(cv_lasso)
cv_lasso

# View coefficients
coef(cv_lasso, s = cv_lasso$lambda.1se) %>% 
  round(4)

# ====================================
# Dimension reduction with PCA
# ====================================

# Prepare data for PCA
trainPredictors = train %>% 
  dplyr::select(-"CTR", -"id")

testPredictors = test %>% 
  dplyr::select(-"id")

# Perform PCA
x = preProcess(x = trainPredictors,method = 'pca',thresh = 0.8)
x

# Create principal components
train_components2 = predict(x,newdata=trainPredictors)
train_components2$CTR = train$CTR

# Fit model on principal components
train_model2 = lm(CTR~.,train_components2)
summary(train_model2)

# Transform test data
test_components2 = predict(x,newdata=testPredictors)
test_components2$CTR = test$CTR

# Calculate test performance
pred = predict(train_model2,newdata=test_components2)
sse = sum((pred-test_components2$CTR)^2)
sst = sum((mean(train_components2$CTR) - test_components2$CTR)^2)
r2_test_model2 = 1 - sse/sst
r2_test_model2

# ====================================
# Final model with selected features
# ====================================

# Double check names in train to input in model
names(train)

# Build final model based on best subset selection results
linearpoly_model <- lm(CTR ~ poly(targeting_score, 2) + 
                       poly(visual_appeal, 2) + 
                       poly(headline_length, 2) + 
                       poly(cta_strength, 2) + 
                       ad_frequency +
                       market_saturation +
                       headline_sentiment +
                       headline_word_count +
                       body_text_length +
                       body_word_count +
                       body_sentiment +
                       body_keyword_density +
                       body_readability_score +
                       brand_familiarity + 
                       contextual_relevance_X1 + 
                       position_on_page_Side.Banner + 
                       position_on_page_Top.Banner +
                       ad_format_Text + 
                       ad_format_Video +
                       age_group_X35.44 + age_group_X25.34 +
                       age_group_X45.54 + age_group_X65.74 +
                       age_group_X55.64 + age_group_X75.84 +
                       age_group_X85. + 
                       gender_Male + gender_Other + 
                       location_South + location_West + 
                       location_Northeast + 
                       time_of_day_Morning + time_of_day_Evening + 
                       time_of_day_Night + 
                       day_of_week_Wednesday + day_of_week_Thursday + 
                       day_of_week_Tuesday + day_of_week_Monday + 
                       day_of_week_Sunday + day_of_week_Saturday + 
                       device_type_Mobile + device_type_Tablet + 
                       seasonality_X1 + 
                       headline_power_words_X1 + 
                       headline_question_X1 + 
                       headline_numbers_X1, 
                     data=train)

# Check for multicollinearity
vif(linearpoly_model)

# Model summary
summary(linearpoly_model)

# ====================================
# Final model evaluation
# ====================================

# Calculate training metrics
pred = predict(linearpoly_model, newdata = train)
sse14 = sum((pred - train$CTR)^2)
sst14 = sum((mean(train$CTR)-train$CTR)^2)
model14_r2 = 1 - sse14/sst14
model14_r2

# Calculate RMSE for training data
rmse14 = sqrt(mean((pred - train$CTR)^2))
rmse14

# Calculate RMSE for test data
pred_test = predict(linearpoly_model, newdata = test)
rmse14_test = sqrt(mean((pred_test - test$CTR)^2))
rmse14_test

# ====================================
# Generate predictions and save results
# ====================================
  
# Generate predictions for full dataset
predictions <- predict(linearpoly_model, analysis_data_transformed)

# Create submission with all predictions
submission <- data.frame(
  id = analysis_data$id,
  CTR = predictions
)

# Save with relative path
write.csv(submission, "output/prediction_linearpoly.csv", row.names = FALSE)

# Create output directories
dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)

# Save EDA plots
ggsave("output/figures/numeric_distributions.png", numeric_plot, width=12, height=8)
ggsave("output/figures/categorical_distributions.png", categorical_plot, width=12, height=8)
ggsave("output/figures/transformed_distributions.png", analysis_data_plot, width=12, height=8)
ggsave("output/figures/ctr_outliers.png", CTR_plot, width=8, height=6)

# Save analysis plots  
ggsave("output/figures/correlation_matrix.png", corplot, width=10, height=8)
ggsave("output/figures/vif_analysis.png", vif_plot, width=10, height=8)
ggsave("output/figures/subset_selection.png", subset_selection_plot, width=10, height=8)
ggsave("output/figures/stepwise_selection.png", stepwise_plot, width=10, height=6)
ggsave("output/figures/backward_selection.png", backward_plot, width=10, height=6)
ggsave("output/figures/hybrid_selection.png", hybridstep_plot, width=10, height=6)
ggsave("output/figures/ridge_paths.png", ridge_plot, width=10, height=6)
