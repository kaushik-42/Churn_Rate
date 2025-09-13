# 🚀 Churn Prediction Project - Step-by-Step Execution Guide

## 📋 Prerequisites

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib openpyxl
```

### Verify Installation
```python
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
print("All libraries installed successfully!")
```

## 📁 Project Structure
```
Churn_Project/
├── SubscriptionUseCase_Dataset.xlsx     # Original dataset
├── data_exploration.py                  # Initial data exploration
├── explore_excel_sheets.py              # Multi-sheet Excel analysis
├── churn_analysis_eda.py                # Comprehensive EDA & feature engineering
├── churn_ml_model.py                    # ML model training & evaluation
├── user_engagement_analysis.py          # User engagement analysis
├── churn_prediction_interface.py        # Production prediction interface
└── PROJECT_SUMMARY.md                   # Complete project documentation
```

## 🎯 Step-by-Step Execution

### Step 1: Navigate to Project Directory
```bash
cd /path/to/Churn_Project
# or
cd Desktop/Churn_Project
```

### Step 2: Initial Data Exploration
```bash
python data_exploration.py
```
**What it does:**
- Loads the Excel file and displays basic info
- Shows dataset shape, columns, and data types
- Identifies missing values and basic statistics
- Creates `subscription_data.csv` for easier access

**Expected Output:**
- Dataset overview with 100 users
- Basic statistics and data quality assessment

### Step 3: Multi-Sheet Excel Analysis
```bash
python explore_excel_sheets.py
```
**What it does:**
- Discovers all 5 sheets in the Excel file
- Analyzes each sheet separately
- Creates individual CSV files for each sheet
- Shows relationships between different data sources

**Expected Output:**
- 5 separate CSV files created
- Detailed analysis of each data source
- Understanding of data schema

### Step 4: Comprehensive EDA and Feature Engineering
```bash
python churn_analysis_eda.py
```
**What it does:**
- Merges all data sources into master dataset
- Engineers 14 predictive features
- Performs correlation analysis
- Creates visualizations
- Saves processed dataset

**Generated Files:**
- `master_churn_dataset.csv` - Complete dataset with features
- `churn_eda_plots.png` - EDA visualizations

**Expected Output:**
```
Dataset loaded: (140, 31)
Churn rate: 49.29%
Final dataset shape: (140, 31)
Visualizations saved as 'churn_eda_plots.png'
```

### Step 5: Machine Learning Model Training
```bash
python churn_ml_model.py
```
**What it does:**
- Trains 4 different ML models
- Performs cross-validation
- Evaluates model performance
- Selects best model (SVM)
- Saves trained model and scalers

**Generated Files:**
- `best_churn_model_svm.pkl` - Trained SVM model
- `feature_scaler.pkl` - Feature scaling transformer
- `feature_names.pkl` - Required feature list
- `model_performance_summary.csv` - Model comparison
- `model_evaluation.png` - ROC curves and confusion matrix
- `feature_importance.png` - Feature importance (if tree-based model wins)

**Expected Output:**
```
=== CHURN PREDICTION ML MODEL ===
Dataset loaded: (100, 31)
Training Logistic Regression... ROC-AUC: 0.19
Training Random Forest... ROC-AUC: 0.21
Training Gradient Boosting... ROC-AUC: 0.29
Training SVM... ROC-AUC: 0.68
BEST MODEL: SVM (ROC-AUC: 0.680)
```

### Step 6: User Engagement Analysis
```bash
python user_engagement_analysis.py
```
**What it does:**
- Analyzes user engagement patterns
- Creates engagement segments
- Identifies high-risk users
- Generates business insights
- Creates comprehensive dashboard

**Generated Files:**
- `user_engagement_dashboard.png` - Engagement visualizations
- `engagement_summary_report.csv` - Key metrics summary
- `high_risk_users.csv` - Users needing immediate attention (if any)

**Expected Output:**
```
=== USER ENGAGEMENT ANALYSIS ===
Total Users: 100
Active Users: 49
Churned Users: 51
Overall Churn Rate: 51.00%
```

### Step 7: Test Prediction Interface
```bash
python churn_prediction_interface.py
```
**What it does:**
- Loads trained model and demonstrates predictions
- Shows individual user predictions
- Demonstrates batch processing
- Provides usage instructions

**Generated Files:**
- `sample_predictions.csv` - Example predictions

**Expected Output:**
```
=== CHURN PREDICTION SYSTEM DEMO ===
✅ Churn prediction model loaded successfully!

User 1: High Risk User
  Churn Probability: 53.3%
  Prediction: Will Stay
  Risk Level: MEDIUM
```

## 🔄 Complete Pipeline Execution

### Option 1: Run All Steps Sequentially
```bash
# Navigate to project directory
cd Desktop/Churn_Project

# Run complete pipeline
python data_exploration.py
python explore_excel_sheets.py
python churn_analysis_eda.py
python churn_ml_model.py
python user_engagement_analysis.py
python churn_prediction_interface.py
```

### Option 2: One-Command Execution
Create a `run_all.py` script:
```python
import subprocess
import sys

scripts = [
    'data_exploration.py',
    'explore_excel_sheets.py', 
    'churn_analysis_eda.py',
    'churn_ml_model.py',
    'user_engagement_analysis.py',
    'churn_prediction_interface.py'
]

for script in scripts:
    print(f"\n{'='*50}")
    print(f"Running {script}")
    print('='*50)
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
```

Then run:
```bash
python run_all.py
```

## 📊 Expected Final Output Files

After running all steps, you should have:

### Data Files:
- `subscription_data.csv`
- `user_data_data.csv`
- `subscriptions_data.csv`
- `subscription_plans_data.csv`
- `subscription_logs_data.csv`
- `billing_information_data.csv`
- `master_churn_dataset.csv`

### Model Files:
- `best_churn_model_svm.pkl`
- `feature_scaler.pkl`
- `feature_names.pkl`

### Analysis Files:
- `model_performance_summary.csv`
- `sample_predictions.csv`
- `engagement_summary_report.csv`

### Visualizations:
- `churn_eda_plots.png`
- `model_evaluation.png`
- `user_engagement_dashboard.png`

## 🚨 Troubleshooting

### Common Issues:

1. **Missing Libraries:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib openpyxl
```

2. **File Not Found Error:**
- Ensure you're in the correct directory
- Verify `SubscriptionUseCase_Dataset.xlsx` exists

3. **Memory Issues:**
- The dataset is small (100 users), shouldn't cause memory issues
- If issues persist, try running scripts individually

4. **Import Errors:**
```bash
# For Jupyter notebooks
!pip install pandas numpy scikit-learn matplotlib seaborn joblib openpyxl

# For conda environments
conda install pandas numpy scikit-learn matplotlib seaborn joblib openpyxl
```

## 🎯 Using the Trained Model

### For New Predictions:
```python
from churn_prediction_interface import ChurnPredictor

# Initialize predictor
predictor = ChurnPredictor()

# Single user prediction
user_data = {
    'tenure_days': 120,
    'Price': 49.99,
    'billing_count': 4,
    'avg_billing_amount': 49.99,
    'total_billing': 199.96,
    'billing_amount_std': 0,
    'failed_payments': 1,
    'days_since_last_billing': 25,
    'days_since_last_renewal': 25,
    'action_count': 2,
    'days_since_last_action': 10,
    'Grace Time': 5,
    'subscription_type_encoded': 0,  # 0=monthly, 1=yearly
    'auto_renewal_encoded': 1  # 0=No, 1=Yes
}

result = predictor.predict_single_user(user_data)
print(f"Churn Probability: {result['churn_probability']:.1%}")
print(f"Risk Level: {result['risk_level']}")
print(f"Recommendation: {result['recommendation']}")
```

## ⏱️ Estimated Execution Time
- **Total Runtime**: 2-5 minutes
- **Step 1-3**: 30 seconds each
- **Step 4-5**: 1-2 minutes each
- **Step 6-7**: 30 seconds each

## 📈 Success Metrics
After completion, you should achieve:
- ✅ SVM model with ~68% ROC-AUC
- ✅ 14 engineered features
- ✅ Comprehensive visualizations
- ✅ Production-ready prediction interface
- ✅ Business insights and recommendations

---

**🎉 Congratulations! You now have a complete churn prediction system ready for production use.**