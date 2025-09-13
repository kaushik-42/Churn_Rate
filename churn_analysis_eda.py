import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

print("Loading all datasets...")

# Load all datasets
user_data = pd.read_csv('user_data_data.csv')
subscriptions = pd.read_csv('subscriptions_data.csv')
subscription_plans = pd.read_csv('subscription_plans_data.csv')
subscription_logs = pd.read_csv('subscription_logs_data.csv')
billing_info = pd.read_csv('billing_information_data.csv')

print("=== COMPREHENSIVE DATA ANALYSIS ===\n")

# Clean billing data (remove unnamed columns)
billing_info = billing_info.drop(['Unnamed: 5', 'Unnamed: 6'], axis=1)

print("1. DATASET OVERVIEW")
print(f"Users: {len(user_data)}")
print(f"Subscriptions: {len(subscriptions)}")
print(f"Subscription Plans: {len(subscription_plans)}")
print(f"Subscription Logs: {len(subscription_logs)}")
print(f"Billing Records: {len(billing_info)}")

print("\n2. USER STATUS DISTRIBUTION")
print(user_data['Status'].value_counts())
print(f"User churn rate: {(user_data['Status'] == 'inactive').mean():.2%}")

print("\n3. SUBSCRIPTION ANALYSIS")
print("\nSubscription Types:")
print(subscriptions['Subscription Type'].value_counts())

print("\nSubscription Status:")
print(subscriptions['Status'].value_counts())

print("\n4. BILLING ANALYSIS")
print("\nPayment Status Distribution:")
print(billing_info['payment_status'].value_counts())

print(f"\nAverage billing amount: ${billing_info['amount'].mean():.2f}")
print(f"Total revenue: ${billing_info['amount'].sum():,.2f}")

print("\n5. SUBSCRIPTION PLANS")
print("\nPlan pricing statistics:")
print(subscription_plans['Price'].describe())

print("\nAuto renewal distribution:")
print(subscription_plans['Auto Renewal Allowed'].value_counts())

# Create comprehensive analysis
print("\n=== CREATING CHURN ANALYSIS FEATURES ===")

# Convert date columns
date_columns = ['Start Date', 'Last Billed Date', 'Last Renewed Date']
for col in date_columns:
    subscriptions[col] = pd.to_datetime(subscriptions[col])

billing_info['billing_date'] = pd.to_datetime(billing_info['billing_date'])
subscription_logs['action date'] = pd.to_datetime(subscription_logs['action date'])

# Calculate features for each user
current_date = datetime.now()

# Merge data to create master dataset
master_data = user_data.merge(subscriptions, on='User Id', how='left', suffixes=('_user', '_sub'))
master_data = master_data.merge(subscription_plans, on='Product Id', how='left')

# Feature engineering
print("\nEngineering features...")

# 1. Tenure (days since subscription start)
master_data['tenure_days'] = (current_date - master_data['Start Date']).dt.days

# 2. Days since last billing
master_data['days_since_last_billing'] = (current_date - master_data['Last Billed Date']).dt.days

# 3. Days since last renewal
master_data['days_since_last_renewal'] = (current_date - master_data['Last Renewed Date']).dt.days

# 4. Billing frequency and amount per user
billing_summary = billing_info.groupby('subscription_id').agg({
    'amount': ['count', 'mean', 'sum', 'std'],
    'payment_status': lambda x: (x == 'failed').sum()
}).round(2)

billing_summary.columns = ['billing_count', 'avg_billing_amount', 'total_billing', 'billing_amount_std', 'failed_payments']
billing_summary = billing_summary.reset_index()

master_data = master_data.merge(billing_summary, left_on='Subscription Id', right_on='subscription_id', how='left')

# 5. Subscription activity from logs
activity_summary = subscription_logs.groupby('Subscription id').agg({
    'action': 'count',
    'action date': 'max'
}).reset_index()
activity_summary.columns = ['Subscription Id', 'action_count', 'last_action_date']
activity_summary['days_since_last_action'] = (current_date - pd.to_datetime(activity_summary['last_action_date'])).dt.days

master_data = master_data.merge(activity_summary, on='Subscription Id', how='left')

# Fill missing values
master_data = master_data.fillna(0)

# Create churn target variable (inactive users are churned)
master_data['churned'] = (master_data['Status_user'] == 'inactive').astype(int)

print(f"\nFinal dataset shape: {master_data.shape}")
print(f"Churn rate: {master_data['churned'].mean():.2%}")

# Save the processed dataset
master_data.to_csv('master_churn_dataset.csv', index=False)
print("\nMaster dataset saved as 'master_churn_dataset.csv'")

# Display key features
key_features = ['User Id', 'churned', 'tenure_days', 'Price', 'billing_count', 
                'avg_billing_amount', 'failed_payments', 'days_since_last_billing',
                'action_count', 'Auto Renewal Allowed']

print("\nKey features for churn prediction:")
print(master_data[key_features].head(10))

print("\n=== BASIC VISUALIZATIONS ===")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Churn distribution
axes[0,0].pie(master_data['churned'].value_counts(), labels=['Active', 'Churned'], autopct='%1.1f%%')
axes[0,0].set_title('Churn Distribution')

# 2. Price distribution by churn
sns.boxplot(data=master_data, x='churned', y='Price', ax=axes[0,1])
axes[0,1].set_title('Price Distribution by Churn Status')

# 3. Tenure vs Churn
sns.boxplot(data=master_data, x='churned', y='tenure_days', ax=axes[1,0])
axes[1,0].set_title('Tenure (Days) by Churn Status')

# 4. Failed payments vs Churn
sns.boxplot(data=master_data, x='churned', y='failed_payments', ax=axes[1,1])
axes[1,1].set_title('Failed Payments by Churn Status')

plt.tight_layout()
plt.savefig('visualizations/churn_eda_plots.png', dpi=300, bbox_inches='tight')
print("Visualizations saved as 'visualizations/churn_eda_plots.png'")

# Feature correlation with churn
print("\n=== FEATURE CORRELATION WITH CHURN ===")
numeric_features = master_data.select_dtypes(include=[np.number]).columns
correlation_with_churn = master_data[numeric_features].corrwith(master_data['churned']).sort_values(ascending=False)
print("Top correlations with churn:")
print(correlation_with_churn.head(10))
print("\nBottom correlations with churn:")
print(correlation_with_churn.tail(5))