import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self):
        """Initialize the churn prediction interface"""
        try:
            self.model = joblib.load('best_churn_model_svm.pkl')
            self.scaler = joblib.load('feature_scaler.pkl')
            self.feature_names = joblib.load('feature_names.pkl')
            print("✅ Churn prediction model loaded successfully!")
        except FileNotFoundError:
            print("❌ Model files not found. Please run the training script first.")
            return
    
    def predict_single_user(self, user_data):
        """
        Predict churn probability for a single user
        
        Parameters:
        user_data (dict): Dictionary containing user features
        
        Returns:
        dict: Prediction results with probability and risk level
        """
        try:
            # Create DataFrame from user data
            df = pd.DataFrame([user_data])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Select and order features
            X = df[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            churn_probability = self.model.predict_proba(X_scaled)[0, 1]
            prediction = self.model.predict(X_scaled)[0]
            
            # Determine risk level
            if churn_probability >= 0.7:
                risk_level = "HIGH"
            elif churn_probability >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                'churn_probability': round(churn_probability, 3),
                'prediction': 'Will Churn' if prediction == 1 else 'Will Stay',
                'risk_level': risk_level,
                'recommendation': self._get_recommendation(churn_probability, user_data)
            }
            
        except Exception as e:
            return {'error': f"Prediction failed: {str(e)}"}
    
    def _get_recommendation(self, probability, user_data):
        """Generate personalized recommendations based on churn probability and user characteristics"""
        
        if probability >= 0.7:
            return "URGENT: High churn risk. Immediate intervention needed. Consider personal outreach, special offers, or account review."
        elif probability >= 0.4:
            return "MODERATE: Monitor closely. Consider engagement campaigns, feature tutorials, or customer satisfaction surveys."
        else:
            return "LOW RISK: Continue standard engagement. Monitor for any changes in behavior patterns."
    
    def predict_batch(self, df):
        """
        Predict churn for multiple users
        
        Parameters:
        df (DataFrame): DataFrame containing user features
        
        Returns:
        DataFrame: Original data with prediction results
        """
        try:
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Select and order features
            X = df[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            predictions = self.model.predict(X_scaled)
            
            # Add results to dataframe
            df['churn_probability'] = probabilities
            df['churn_prediction'] = ['Will Churn' if p == 1 else 'Will Stay' for p in predictions]
            df['risk_level'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW' for p in probabilities]
            
            return df
            
        except Exception as e:
            print(f"Batch prediction failed: {str(e)}")
            return None

def demo_predictions():
    """Demonstrate the churn prediction system with sample data"""
    
    print("=== CHURN PREDICTION SYSTEM DEMO ===\n")
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Sample user profiles for demonstration
    sample_users = [
        {
            'user_id': 1001,
            'name': 'High Risk User',
            'tenure_days': 30,
            'Price': 25.99,
            'billing_count': 1,
            'avg_billing_amount': 25.99,
            'total_billing': 25.99,
            'billing_amount_std': 0,
            'failed_payments': 2,
            'days_since_last_billing': 45,
            'days_since_last_renewal': 45,
            'action_count': 0,
            'days_since_last_action': 30,
            'Grace Time': 5,
            'subscription_type_encoded': 0,  # monthly
            'auto_renewal_encoded': 0  # No
        },
        {
            'user_id': 1002,
            'name': 'Medium Risk User',
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
            'subscription_type_encoded': 0,  # monthly
            'auto_renewal_encoded': 1  # Yes
        },
        {
            'user_id': 1003,
            'name': 'Low Risk User',
            'tenure_days': 365,
            'Price': 99.99,
            'billing_count': 12,
            'avg_billing_amount': 99.99,
            'total_billing': 1199.88,
            'billing_amount_std': 0,
            'failed_payments': 0,
            'days_since_last_billing': 15,
            'days_since_last_renewal': 15,
            'action_count': 8,
            'days_since_last_action': 2,
            'Grace Time': 5,
            'subscription_type_encoded': 1,  # yearly
            'auto_renewal_encoded': 1  # Yes
        }
    ]
    
    print("1. INDIVIDUAL USER PREDICTIONS\n")
    
    for i, user in enumerate(sample_users, 1):
        print(f"User {i}: {user['name']}")
        result = predictor.predict_single_user(user)
        
        if 'error' not in result:
            print(f"  Churn Probability: {result['churn_probability']:.1%}")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Recommendation: {result['recommendation']}")
        else:
            print(f"  Error: {result['error']}")
        print("-" * 50)
    
    # Batch prediction demo
    print("\n2. BATCH PREDICTION\n")
    
    sample_df = pd.DataFrame(sample_users)
    batch_results = predictor.predict_batch(sample_df)
    
    if batch_results is not None:
        print("Batch Prediction Results:")
        display_columns = ['name', 'churn_probability', 'churn_prediction', 'risk_level']
        print(batch_results[display_columns].to_string(index=False))
        
        # Save results
        batch_results.to_csv('sample_predictions.csv', index=False)
        print("\nBatch results saved as 'sample_predictions.csv'")
    
    print("\n3. FEATURE IMPORTANCE (for new predictions)")
    print("Required features for prediction:")
    for i, feature in enumerate(predictor.feature_names, 1):
        print(f"{i:2d}. {feature}")
    
    print("\n4. USAGE INSTRUCTIONS")
    print("""
To use this prediction system:

1. For single predictions:
   predictor = ChurnPredictor()
   result = predictor.predict_single_user(user_data_dict)

2. For batch predictions:
   predictor = ChurnPredictor()
   results_df = predictor.predict_batch(your_dataframe)

3. Required features for prediction:
   - All 14 features listed above must be present
   - Missing features will be filled with 0
   - Categorical features must be encoded as numbers

4. Interpretation:
   - Probability > 70%: HIGH risk (immediate action needed)
   - Probability 40-70%: MEDIUM risk (monitor and engage)
   - Probability < 40%: LOW risk (standard engagement)
""")

if __name__ == "__main__":
    demo_predictions()