# 🎯 Customer Churn Prediction ML Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning project that predicts customer churn for telecom companies using the Telco Customer Churn dataset. The project includes end-to-end ML pipeline implemented in Jupyter notebooks from data exploration to production-ready prediction interface.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/kaushik-42/Churn_Rate.git
cd Churn_Rate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter

# Start Jupyter Notebook
jupyter notebook

# Run the notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_feature_engineering.ipynb  
# 3. notebooks/03_exploratory_data_analysis.ipynb
# 4. notebooks/04_machine_learning_models.ipynb
# 5. notebooks/05_prediction_interface.ipynb
```

## 📊 Project Overview

This project develops a machine learning model to predict customer churn using the industry-standard Telco Customer Churn dataset with **84.48% ROC-AUC accuracy** using Logistic Regression. The model identifies customers at risk of churning, enabling proactive retention strategies and business insights.

### Key Features
- 🔍 **Comprehensive EDA** with multi-source data analysis
- 🛠 **Feature Engineering** with 14 predictive features
- 🤖 **ML Model Comparison** (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- 📈 **User Engagement Analysis** with risk segmentation
- 🎯 **Production-Ready Interface** for real-time predictions
- 📊 **Business Insights** and retention recommendations

## 📁 Dataset

The project uses the **Telco Customer Churn Dataset** from Kaggle:
- **Source**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Description**: Comprehensive telecom customer data with churn labels

**Dataset Features:**
- **Customer Demographics**: Gender, age, senior citizen status, partner, dependents
- **Account Information**: Tenure, contract type, payment method, billing preferences
- **Services**: Phone service, internet service, online security, tech support, etc.
- **Billing**: Monthly charges, total charges, paperless billing
- **Target**: Churn (Yes/No)

**Key Statistics:**
- 📊 Total Customers: 7,043
- 📉 Churn Rate: ~27%
- 🏢 Industry: Telecommunications
- 📊 Features: 20 predictive features

## 🛠 Technical Architecture

### Machine Learning Pipeline
```
Raw Data → EDA → Feature Engineering → Model Training → Evaluation → Deployment
```

### Models Tested
| Model | ROC-AUC | Performance |
|-------|---------|-------------|
| **Logistic Regression** | **0.8448** | **🏆 Best Model** |
| Random Forest | 0.82 | Strong Performance |
| Gradient Boosting | 0.81 | Good Accuracy |
| SVM | 0.79 | Good Generalization |

### Key Features Used
1. **Demographics**: `SeniorCitizen`, `Partner`, `Dependents`
2. **Account**: `tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`
3. **Services**: `PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`
4. **Financial**: `MonthlyCharges`, `TotalCharges`
5. **Engineered**: `CLV_estimate`, `services_count`, `avg_monthly_charges`

## 📈 Results & Impact

### Model Performance
- **Best Model**: Logistic Regression (ROC-AUC: 0.8448)
- **F1-Score**: 0.5904
- **Accuracy**: 80.7%
- **Churn Detection Rate**: 52.4% of actual churners identified
- **Cross-Validation**: Robust 5-fold validation
- **Risk Classification**: HIGH (>0.7), MEDIUM (0.4-0.7), LOW (<0.4)

### Business Insights
- 🚨 **27% churn rate** in telecom industry
- 📅 **Month-to-month contracts** have highest churn risk
- 💰 **Higher monthly charges** correlate with increased churn
- 🎯 **New customers** (low tenure) are most at risk
- 📱 **Fiber optic** customers show higher churn tendency
- 👥 **Senior citizens** have different churn patterns

### Business Impact
- 🎯 **52.4% churn detection rate** - identifies half of at-risk customers
- 📊 **1,409 test customers** with 374 actual churners
- ✅ **196 correctly identified** high-risk customers for targeted campaigns
- 💰 **Revenue protection** through proactive retention
- ⚡ **Automated risk scoring** reduces manual analysis
- 🎯 **Data-driven insights** for retention strategies

## 🔧 Usage

### Single Customer Prediction
```python
from src.telco_churn_predictor import TelcoChurnPredictor

predictor = TelcoChurnPredictor()
customer_data = {
    'tenure': 12,
    'MonthlyCharges': 75.50,
    'TotalCharges': 906.00,
    'Contract': 'Month-to-month',
    'PaymentMethod': 'Electronic check',
    'InternetService': 'Fiber optic',
    # ... other features
}

result = predictor.predict_single_customer(customer_data)
print(f"Churn Probability: {result['churn_probability']:.1%}")
print(f"Risk Level: {result['risk_level']}")
print(f"Model: Logistic Regression (ROC-AUC: 0.8448)")
```

### Batch Processing
```python
# Load your customer dataset
df = pd.read_csv('your_customer_data.csv')
results = predictor.predict_batch(df)

# Get high-risk customers for targeted campaigns
high_risk = results[results['risk_level'] == 'HIGH']
```

## 📊 Visualizations

The project generates comprehensive visualizations:

- 📈 **EDA Plots**: Distribution analysis and correlations
- 🎯 **Model Evaluation**: ROC curves and confusion matrices
- 👥 **User Engagement Dashboard**: Behavior analysis
- 📊 **Feature Importance**: Key predictive factors

## 📋 File Structure

```
Churn_Project/
├── 📓 Jupyter Notebooks
│   ├── 01_data_exploration.ipynb           # Initial data exploration
│   ├── 02_feature_engineering.ipynb        # Data preprocessing & feature engineering
│   ├── 03_exploratory_data_analysis.ipynb  # Comprehensive EDA
│   ├── 04_machine_learning_models.ipynb    # Model training & evaluation
│   └── 05_prediction_interface.ipynb       # Production interface
├── 📊 Data Files
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Telco dataset
│   └── processed_data/                      # Processed datasets
├── 🤖 Model Files
│   ├── models/                             # Trained models
│   ├── scalers/                            # Feature scalers
│   └── encoders/                           # Label encoders
├── 🔍 Source Code
│   └── src/telco_churn_predictor.py        # Production prediction class
├── 📊 Visualizations
│   └── visualizations/                     # All plots and charts
├── 📖 Documentation
│   ├── README.md                           # This file
│   └── docs/                               # Additional documentation
└── ⚙️ Configuration
    └── .gitignore                          # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, openpyxl

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/churn-prediction-ml.git
   cd churn-prediction-ml
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline**
   ```bash
   # Option 1: Run all scripts sequentially
   bash run_pipeline.sh
   
   # Option 2: Run individual scripts
   python churn_ml_model.py
   python churn_prediction_interface.py
   ```

### Expected Runtime
- **Total Execution**: 2-5 minutes
- **Model Training**: 1-2 minutes
- **Analysis & Visualization**: 1-2 minutes

## 📖 Documentation

- 📋 **[RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md)**: Complete step-by-step execution guide
- 📊 **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: Detailed analysis and findings
- 🤖 **Code Documentation**: Inline comments and docstrings

## 🎯 Business Recommendations

### Immediate Actions
1. 🎯 **Target Month-to-Month Customers**: Convert to longer contracts
2. 📞 **Proactive Support**: Reach out to fiber optic customers
3. 💰 **Pricing Review**: Optimize pricing for high-charge customers
4. 👥 **Senior Citizen Programs**: Tailored retention for seniors

### Long-term Strategies
1. 🏆 **Contract Incentives**: Promote longer-term contracts
2. ⭐ **Service Bundling**: Increase service adoption
3. 🔔 **Predictive Interventions**: Automated risk monitoring
4. 📊 **Data-Driven Pricing**: Dynamic pricing based on churn risk

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Kaushik Tummalapalli**
- 📧 Email: [your.email@example.com]
- 💼 LinkedIn: [Your LinkedIn Profile]
- 🐙 GitHub: [@yourusername]

## 🙏 Acknowledgments

- Thanks to the open-source community for amazing ML libraries
- scikit-learn for comprehensive machine learning tools
- pandas and numpy for data manipulation
- matplotlib and seaborn for beautiful visualizations

---

⭐ **Star this repo if you found it helpful!** ⭐

![Churn Prediction](https://img.shields.io/badge/Churn%20Prediction-ML%20Project-brightgreen)
![Data Science](https://img.shields.io/badge/Data%20Science-Customer%20Analytics-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SVM-orange)