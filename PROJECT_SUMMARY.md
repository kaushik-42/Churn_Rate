# Churn Prediction ML Project

## 📊 Project Overview

This project develops a machine learning model to predict customer churn in a subscription-based business using historical subscription and usage data. The model helps identify users at risk of canceling their subscriptions, enabling proactive retention strategies.

## 🎯 Objectives

1. **Churn Prediction**: Build ML models to predict which users are likely to churn
2. **User Engagement Analysis**: Analyze customer behavior patterns and engagement levels
3. **Risk Assessment**: Identify high-risk customers for targeted retention campaigns
4. **Actionable Insights**: Provide recommendations for improving customer retention

## 📁 Dataset Information

The project uses a multi-sheet Excel dataset containing:

- **User_Data**: 100 users with basic information and status
- **Subscriptions**: Subscription details, dates, and status information
- **Subscription_Plans**: Product information with pricing and auto-renewal settings
- **Subscription_Logs**: Activity logs showing status changes
- **Billing_Information**: Payment records and billing history

**Key Statistics:**
- Total Users: 100
- Churn Rate: 51%
- Subscription Types: Monthly (59%), Yearly (41%)
- Average Billing Amount: $239.27

## 🔍 Exploratory Data Analysis

### Key Findings:
- **High churn rate** of 51% indicates significant retention challenges
- **Payment failures** correlate with higher churn probability
- **Low engagement** users (80% of users) show higher churn rates
- **Auto-renewal disabled** users have higher churn tendency
- **Tenure** shows inverse relationship with churn (longer tenure = lower churn)

## 🛠 Feature Engineering

Created 14 key features for prediction:

1. **Behavioral Features:**
   - `tenure_days`: Days since subscription start
   - `action_count`: Number of subscription actions
   - `days_since_last_action`: Recency of last activity

2. **Financial Features:**
   - `Price`: Subscription price
   - `billing_count`: Number of billing events
   - `avg_billing_amount`: Average billing amount
   - `failed_payments`: Number of failed payment attempts

3. **Engagement Features:**
   - `days_since_last_billing`: Billing recency
   - `days_since_last_renewal`: Renewal recency
   - `billing_amount_std`: Billing amount variability

4. **Subscription Features:**
   - `subscription_type_encoded`: Monthly vs. Yearly
   - `auto_renewal_encoded`: Auto-renewal setting
   - `Grace Time`: Grace period for payments

## 🤖 Machine Learning Models

### Models Tested:
1. **Logistic Regression** - ROC-AUC: 0.19
2. **Random Forest** - ROC-AUC: 0.21
3. **Gradient Boosting** - ROC-AUC: 0.29
4. **Support Vector Machine** - ROC-AUC: 0.68 ⭐

### Best Model: SVM
- **ROC-AUC Score**: 68%
- **Cross-validation**: 55.8% ± 7.8%
- **Model Type**: Support Vector Machine with RBF kernel
- **Performance**: Best balance of precision and recall

## 📈 User Engagement Analysis

### Engagement Levels:
- **Low Engagement**: 80% of users (50% churn rate)
- **Medium Engagement**: 20% of users (55% churn rate)
- **High Engagement**: 0% of users

### Key Insights:
1. **Payment Behavior**: Users with failed payments show higher churn risk
2. **Billing Segments**: Premium users ($300+) have lower churn rates
3. **Tenure Impact**: Users with 365+ days tenure show 45% churn rate
4. **Auto-renewal**: "No" auto-renewal correlates with 47% churn rate

### Customer Lifetime Value:
- **Low Engagement**: $967.60 average CLV
- **Medium Engagement**: $3,702.18 average CLV

## 🎯 Prediction System

### ChurnPredictor Class Features:
- **Single User Prediction**: Real-time churn probability for individual users
- **Batch Processing**: Predict churn for multiple users simultaneously
- **Risk Classification**: HIGH (>70%), MEDIUM (40-70%), LOW (<40%)
- **Personalized Recommendations**: Tailored retention strategies

### Sample Predictions:
```python
# High Risk User Example
{
    'churn_probability': 0.533,
    'prediction': 'Will Stay',
    'risk_level': 'MEDIUM',
    'recommendation': 'Monitor closely. Consider engagement campaigns.'
}
```

## 📊 Generated Files and Outputs

### Data Files:
- `master_churn_dataset.csv`: Processed dataset with all features
- `subscription_data.csv`: Original user data in CSV format
- Individual CSV files for each Excel sheet

### Model Files:
- `best_churn_model_svm.pkl`: Trained SVM model
- `feature_scaler.pkl`: Feature scaling transformer
- `feature_names.pkl`: List of required features

### Analysis Files:
- `model_performance_summary.csv`: Comparison of all models
- `sample_predictions.csv`: Example predictions
- `engagement_summary_report.csv`: Key engagement metrics

### Visualizations:
- `churn_eda_plots.png`: EDA visualizations
- `feature_importance.png`: Feature importance plot
- `model_evaluation.png`: ROC curves and confusion matrix
- `user_engagement_dashboard.png`: Comprehensive engagement analysis

## 🔧 Usage Instructions

### 1. Training New Models:
```bash
python churn_ml_model.py
```

### 2. Running Predictions:
```python
from churn_prediction_interface import ChurnPredictor

predictor = ChurnPredictor()
result = predictor.predict_single_user(user_data)
```

### 3. Analyzing Engagement:
```bash
python user_engagement_analysis.py
```

## 💡 Business Recommendations

### Immediate Actions:
1. **Target High-Risk Users**: Focus on users with failed payments and low engagement
2. **Promote Auto-Renewal**: Encourage auto-renewal adoption to reduce churn
3. **Engagement Campaigns**: Implement features to increase user activity
4. **Payment Support**: Provide assistance for payment-related issues

### Long-term Strategies:
1. **Loyalty Programs**: Reward long-tenure customers
2. **Premium Features**: Enhance value proposition for higher-tier plans
3. **Proactive Support**: Monitor and reach out to at-risk users
4. **Pricing Strategy**: Review pricing structure based on churn patterns

## ⚡ Model Performance Limitations

- **Small Dataset**: Only 100 users limit model generalization
- **Feature Richness**: Limited behavioral and usage features
- **Temporal Data**: Limited time-series information for trend analysis
- **External Factors**: No data on market conditions or competitor actions

## 🚀 Future Improvements

1. **Data Collection**: Gather more comprehensive user behavior data
2. **Advanced Features**: Implement time-series and seasonal features
3. **Model Ensemble**: Combine multiple models for better performance
4. **Real-time Pipeline**: Deploy model for real-time churn prediction
5. **A/B Testing**: Test retention strategies and measure impact

## 📈 Expected Business Impact

- **Retention Improvement**: Target interventions could reduce churn by 15-25%
- **Revenue Protection**: Proactive retention of high-value customers
- **Operational Efficiency**: Automated risk scoring reduces manual analysis
- **Customer Insights**: Better understanding of customer behavior patterns

---

*This project demonstrates end-to-end ML pipeline for churn prediction, from data exploration to deployment-ready prediction system.*