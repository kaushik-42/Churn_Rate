import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=== USER ENGAGEMENT ANALYSIS ===\n")

# Load data
df = pd.read_csv('../data/master_churn_dataset.csv')
billing_info = pd.read_csv('../data/billing_information_data.csv')
subscription_logs = pd.read_csv('../data/subscription_logs_data.csv')

# Remove duplicates for user-level analysis
df_users = df.drop_duplicates(subset=['User Id'], keep='first')

print("1. OVERALL ENGAGEMENT METRICS")
print(f"Total Users: {len(df_users)}")
print(f"Active Users: {len(df_users[df_users['churned'] == 0])}")
print(f"Churned Users: {len(df_users[df_users['churned'] == 1])}")
print(f"Overall Churn Rate: {df_users['churned'].mean():.2%}")

# Engagement segments
def categorize_engagement(row):
    if row['action_count'] >= 3 and row['billing_count'] >= 3 and row['failed_payments'] == 0:
        return 'High Engagement'
    elif row['action_count'] >= 1 and row['billing_count'] >= 1 and row['failed_payments'] <= 1:
        return 'Medium Engagement'
    else:
        return 'Low Engagement'

df_users['engagement_level'] = df_users.apply(categorize_engagement, axis=1)

print("\n2. ENGAGEMENT LEVEL DISTRIBUTION")
engagement_dist = df_users['engagement_level'].value_counts()
print(engagement_dist)

# Churn rate by engagement level
print("\n3. CHURN RATE BY ENGAGEMENT LEVEL")
churn_by_engagement = df_users.groupby('engagement_level')['churned'].agg(['count', 'sum', 'mean']).round(3)
churn_by_engagement.columns = ['Total_Users', 'Churned_Users', 'Churn_Rate']
print(churn_by_engagement)

# Subscription type analysis
print("\n4. SUBSCRIPTION TYPE ANALYSIS")
sub_analysis = df_users.groupby('Subscription Type').agg({
    'churned': ['count', 'sum', 'mean'],
    'Price': 'mean',
    'tenure_days': 'mean',
    'billing_count': 'mean'
}).round(2)
print(sub_analysis)

# Billing behavior analysis
print("\n5. BILLING BEHAVIOR ANALYSIS")
billing_segments = pd.cut(df_users['avg_billing_amount'], 
                         bins=[0, 100, 200, 300, float('inf')], 
                         labels=['Low ($0-$100)', 'Medium ($100-$200)', 
                                'High ($200-$300)', 'Premium ($300+)'])

billing_churn = df_users.groupby(billing_segments)['churned'].agg(['count', 'mean']).round(3)
billing_churn.columns = ['User_Count', 'Churn_Rate']
print(billing_churn)

# Payment failure analysis
print("\n6. PAYMENT FAILURE IMPACT")
payment_analysis = df_users.groupby('failed_payments')['churned'].agg(['count', 'mean']).round(3)
payment_analysis.columns = ['User_Count', 'Churn_Rate']
print("Churn rate by number of failed payments:")
print(payment_analysis)

# Tenure analysis
print("\n7. TENURE ANALYSIS")
tenure_segments = pd.cut(df_users['tenure_days'], 
                        bins=[0, 30, 90, 180, 365, float('inf')], 
                        labels=['0-30 days', '30-90 days', '90-180 days', 
                               '180-365 days', '365+ days'])

tenure_churn = df_users.groupby(tenure_segments)['churned'].agg(['count', 'mean']).round(3)
tenure_churn.columns = ['User_Count', 'Churn_Rate']
print("Churn rate by tenure:")
print(tenure_churn)

# Auto-renewal impact
print("\n8. AUTO-RENEWAL IMPACT")
auto_renewal_churn = df_users.groupby('Auto Renewal Allowed')['churned'].agg(['count', 'mean']).round(3)
auto_renewal_churn.columns = ['User_Count', 'Churn_Rate']
print(auto_renewal_churn)

print("\n9. CREATING COMPREHENSIVE VISUALIZATIONS")

# Create comprehensive engagement dashboard
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# 1. Engagement Level Distribution
engagement_dist.plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%')
axes[0,0].set_title('User Engagement Levels')
axes[0,0].set_ylabel('')

# 2. Churn by Engagement Level
churn_by_engagement['Churn_Rate'].plot(kind='bar', ax=axes[0,1], color='coral')
axes[0,1].set_title('Churn Rate by Engagement Level')
axes[0,1].set_ylabel('Churn Rate')
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Subscription Type vs Churn
sub_churn = df_users.groupby('Subscription Type')['churned'].mean()
sub_churn.plot(kind='bar', ax=axes[0,2], color='lightblue')
axes[0,2].set_title('Churn Rate by Subscription Type')
axes[0,2].set_ylabel('Churn Rate')

# 4. Billing Amount Distribution
axes[1,0].hist(df_users['avg_billing_amount'], bins=15, alpha=0.7, color='green')
axes[1,0].set_title('Average Billing Amount Distribution')
axes[1,0].set_xlabel('Average Billing Amount ($)')

# 5. Tenure vs Churn
sns.boxplot(data=df_users, x='churned', y='tenure_days', ax=axes[1,1])
axes[1,1].set_title('Tenure Distribution by Churn Status')
axes[1,1].set_xticklabels(['Active', 'Churned'])

# 6. Failed Payments vs Churn
payment_analysis['Churn_Rate'].plot(kind='bar', ax=axes[1,2], color='red', alpha=0.7)
axes[1,2].set_title('Churn Rate by Failed Payments')
axes[1,2].set_xlabel('Number of Failed Payments')

# 7. Price vs Churn
sns.boxplot(data=df_users, x='churned', y='Price', ax=axes[2,0])
axes[2,0].set_title('Price Distribution by Churn Status')
axes[2,0].set_xticklabels(['Active', 'Churned'])

# 8. Auto Renewal vs Churn
auto_renewal_churn['Churn_Rate'].plot(kind='bar', ax=axes[2,1], color='purple', alpha=0.7)
axes[2,1].set_title('Churn Rate by Auto Renewal Setting')
axes[2,1].tick_params(axis='x', rotation=45)

# 9. Billing Count vs Churn
sns.boxplot(data=df_users, x='churned', y='billing_count', ax=axes[2,2])
axes[2,2].set_title('Billing Count by Churn Status')
axes[2,2].set_xticklabels(['Active', 'Churned'])

plt.tight_layout()
plt.savefig('../visualizations/user_engagement_dashboard.png', dpi=300, bbox_inches='tight')
print("Engagement dashboard saved as '../visualizations/user_engagement_dashboard.png'")

# Customer Lifetime Value (CLV) Analysis
print("\n10. CUSTOMER LIFETIME VALUE ANALYSIS")
df_users['estimated_clv'] = df_users['avg_billing_amount'] * (df_users['tenure_days'] / 30)
clv_by_segment = df_users.groupby('engagement_level')['estimated_clv'].agg(['mean', 'median', 'std']).round(2)
print("CLV by Engagement Level:")
print(clv_by_segment)

# High-risk users identification
print("\n11. HIGH-RISK USERS IDENTIFICATION")
high_risk_criteria = (
    (df_users['failed_payments'] > 0) |
    (df_users['days_since_last_billing'] > 60) |
    (df_users['action_count'] == 0) |
    (df_users['Auto Renewal Allowed'] == 'No')
)

high_risk_users = df_users[high_risk_criteria & (df_users['churned'] == 0)]
print(f"Active users at high risk of churn: {len(high_risk_users)}")
print(f"Percentage of active users at risk: {len(high_risk_users) / len(df_users[df_users['churned'] == 0]):.1%}")

# Save high-risk users for retention campaigns
high_risk_users[['User Id', 'Name_x', 'Email_x', 'failed_payments', 
                'days_since_last_billing', 'action_count', 'Auto Renewal Allowed',
                'estimated_clv']].to_csv('../data/high_risk_users.csv', index=False)

print("\n12. ENGAGEMENT RECOMMENDATIONS")
print("Based on the analysis, here are key recommendations:")
print("1. Focus retention efforts on users with failed payments")
print("2. Implement engagement campaigns for users with low activity")
print("3. Promote auto-renewal options to reduce churn")
print("4. Monitor users who haven't been billed recently")
print("5. Create loyalty programs for long-tenure customers")
print("6. Offer premium features to increase engagement")

# Summary report
summary_report = {
    'Total_Users': len(df_users),
    'Active_Users': len(df_users[df_users['churned'] == 0]),
    'Churned_Users': len(df_users[df_users['churned'] == 1]),
    'Overall_Churn_Rate': f"{df_users['churned'].mean():.2%}",
    'High_Risk_Users': len(high_risk_users),
    'Average_CLV': f"${df_users['estimated_clv'].mean():.2f}",
    'High_Engagement_Users': len(df_users[df_users['engagement_level'] == 'High Engagement']),
    'Low_Engagement_Users': len(df_users[df_users['engagement_level'] == 'Low Engagement'])
}

summary_df = pd.DataFrame(list(summary_report.items()), columns=['Metric', 'Value'])
summary_df.to_csv('../data/engagement_summary_report.csv', index=False)
print("\nSummary report saved as '../data/engagement_summary_report.csv'")
print("\nHigh-risk users saved as '../data/high_risk_users.csv' for targeted retention campaigns")