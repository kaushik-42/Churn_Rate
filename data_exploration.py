import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading the subscription dataset...")
df = pd.read_excel('SubscriptionUseCase_Dataset.xlsx')

print("\n=== DATASET OVERVIEW ===")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n=== FIRST FEW ROWS ===")
print(df.head())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== BASIC STATISTICS ===")
print(df.describe())

print("\n=== UNIQUE VALUES IN CATEGORICAL COLUMNS ===")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(f"Values: {df[col].unique()[:10]}")  # Show first 10 unique values

# Save basic info to CSV for easier examination
df.to_csv('subscription_data.csv', index=False)
print("\nDataset saved as 'subscription_data.csv' for easier access")