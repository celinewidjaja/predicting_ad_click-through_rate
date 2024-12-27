# Digital Advertising Performance Optimization
## Predictive Analytics for Click-Through Rate Enhancement

### Project Overview
This analysis develops a machine learning solution to predict and optimize Click-Through Rates (CTR) for digital advertising campaigns. Using comprehensive campaign data including ad characteristics, audience demographics, and engagement metrics, we've created a predictive model that helps maximize advertising ROI through data-driven targeting and content optimization.

## How to Use This Project
1. Clone this repository: `git clone https://github.com/celinewidjaja/predicting_ad_click-through_rate`
2. Install required dependencies using `requirements.txt` or `environment.yml`:
   - If using pip: `pip install -r requirements.txt`
   - If using conda: `conda env create -f environment.yml`
3. Run any model script from the `src/` folder:
   - Example: `Rscript src/PAC_gam.R`
4. Access outputs (predictions, visualizations) in the `output/` folder.

## Data Structure Overview

#### The dataset analyzes ad campaign performance across multiple dimensions:

1. **Ad Quality Metrics:**  
   - Visual appeal (1-10 scale)  
   - Targeting score (1-10 scale)  
   - CTA strength (1-10 scale)  
   - Contextual relevance (binary)

2. **Content Characteristics:**  
   - Headline length (character count)  
   - Body readability score (1-100)  
   - Ad format (Image, Video, Text)  
   - Position on page

3. **Audience Demographics:**  
   - Age group  
   - Gender  
   - Location (Northeast, Midwest, South, West)

4. **Engagement Context:**  
   - Time of day  
   - Day of week  
   - Device type  
   - Seasonality

## Methodology 

#### **1. Data Preprocessing:**
- Identified missing data, outliers, and out of range data
- Handled missing values using bag imputation (for final model)
- Preliminary models used a mix of bag, mean imputation, and proportion handling
- Treated outliers through capping where appropriate
- Standardized numeric variables
- Created dummy variables for categorical features

#### **2. Feature Selection:**
Applied multiple techniques to identify key predictors:  
- Best subset selection  
- Forward/backward selection  
- LASSO and ridge regression   
- Principal Component Analysis (PCA)  

#### **3. Model Development:**
Progressed through increasingly sophisticated approaches:  
- Linear regression
- Linear regression with polynomial terms
- Generalized Additive Models (GAM) 
- Bagging  
- Random forest  
- XGBoost 

#### **4. Variable Engineering:**
Created composite metrics including:  
- Content quality score  
- Peak hour engagement indicators  
- Temporal pattern indicators  

## Executive Summary
Analysis of the digital advertising campaign data reveals that visual appeal is the strongest predictor of CTR (0.561 correlation), followed by targeting precision (0.394 correlation). Four key variables demonstrate important non-linear relationships with CTR: targeting score, visual appeal, CTA strength, and headline length. Mobile devices dominate traffic (69.7%), with peak engagement during afternoon hours (1,585 impressions). The analysis highlights the importance of optimizing visual content and targeting while considering temporal patterns in engagement. These insights led to a predictive model achieving strong performance (cross-validation RMSE: 0.05903), significantly improving upon baseline predictions.

## Insights Deep Dive

### **1. Key Drivers of Ad Performance**
The analysis revealed visual quality and targeting precision as the primary drivers of CTR. Visual appeal showed the strongest correlation (0.561), indicating it's more crucial than traditionally emphasized factors like ad placement. Targeting score demonstrated the second-highest impact (0.394), but with diminishing returns at higher levels. Both metrics showed non-linear relationships with CTR, suggesting optimal ranges rather than simple linear improvements.

### **2. Content and Format Impact**
Content characteristics showed significant influence on performance:
- Headline length demonstrates an optimal range around 29 characters (median)
- Body content maintains strong readability (mean score: 74.87)
- Video format emerged as a strong performer
- CTA strength shows complex non-linear behavior, requiring careful optimization

### **3. Temporal and Geographic Patterns**
Clear patterns emerged in timing and location:
- Afternoon hours lead with 1,585 impressions
- Morning follows with 1,189 impressions
- Evening (821) and night (405) show lower engagement
- Geographic distribution shows stronger presence in South (1,308 users) and Midwest (894 users)

### **4. Device and Platform Impact**
Device usage patterns reveal clear preferences:
- Mobile dominance: 69.7% of traffic
- Desktop usage: 28.3%
- Tablet engagement: 2%
These patterns suggest the critical importance of mobile optimization.

## Model Development Journey
Through iterative refinement, our modeling approach evolved:

1. **Initial Linear Model**  
   - Starting RMSE: 0.1058603  

2. **Polynomial Linear Model**  
   - After adding polynomial terms to capture non-linear relationships: RMSE improved to 0.06539632  
   
3. **GAM Model for Non-Linear Patterns**  
   - GAM achieved a cross-validation RMSE of 0.06799151.

4. **Tree-Based Models**  
   - **Random Forest**: RMSE of 0.0645 (cross-validation)  
   - **Bagging**: RMSE of 0.0731 (cross-validation)  

5. **XGBoost Model**  
   - Cross-validation RMSE: 0.05903

### Model Comparison Summary
| Model                  | Cross-Validation RMSE |
|------------------------|------------------------|
| Linear Regression      | 0.1058                 |
| Polynomial Linear Model| 0.0654                 |
| GAM                    | 0.0679                 |
| Random Forest          | 0.0645                 |
| Bagging                | 0.0731                 |
| XGBoost                | 0.0590                 |

## Recommendations

1. **Visual Content Strategy**  
   - Prioritize high-resolution and aesthetically pleasing images. Use tools like A/B testing to refine ad visuals and identify the most engaging designs.  
   - Regularly update visuals to maintain audience interest and reduce ad fatigue.  
   - Ensure alignment between visuals and messaging to enhance contextual relevance.
   
2. **Targeting Optimization**  
   - Invest in tools for real-time audience segmentation based on browsing patterns and demographic data.  
   - Maintain targeting precision within the optimal score range (7-9) to maximize CTR without incurring diminishing returns.  
   - Reassess audience definitions quarterly to adjust to shifts in market behavior.

3. **Content Refinement**  
   - Limit headline length to around 29 characters, balancing brevity and clarity. Use strong action verbs to drive engagement.  
   - Employ readability analysis tools to ensure ad copy scores above 70. Test variations of CTAs to identify language that resonates with specific audiences.  
   - Incorporate multimedia formats (e.g., videos) for complex messages or promotions.

4. **Timing Strategy**  
   - Focus on scheduling ads during peak hours (12 PM - 3 PM) for maximum engagement. Use performance analytics to fine-tune these time windows further.  
   - Diversify targeting across high-performing geographic regions, such as the South and Midwest, while testing underperforming regions for potential growth.  
   - Adjust for seasonality by aligning ad themes with relevant holidays or events to maximize contextual engagement.

5. **Timing Strategy**  
   - Xgboost highlights the dynamic nature of variables influencing CTR, suggesting the need for ongoing refinement.  
   - Monitor campaigns regularly and adjust strategies.
   - Implement feedback loops where campaign performance informs future Xgboost training for improved predictive accuracy.
   
## Caveats & Assumptions
The analysis assumes consistent measurement of subjective metrics like visual appeal and targeting score. The model's performance may vary with changes in market conditions or consumer behavior. Geographic and demographic patterns might reflect current campaign targeting rather than inherent preferences. Additional data on user behavior and long-term engagement would strengthen these findings.
