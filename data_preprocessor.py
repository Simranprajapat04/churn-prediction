"""
Data Preprocessing Module
Handles data cleaning, feature engineering, and SMOTE for imbalanced data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles all data preprocessing tasks for churn prediction"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        self.feature_columns = None
        
    def preprocess_data(self, data, apply_smote=True, test_size=0.2):
        """
        Complete preprocessing pipeline
        
        Args:
            data (pd.DataFrame): Raw customer data
            apply_smote (bool): Whether to apply SMOTE for balancing
            test_size (float): Proportion of data for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        print("Starting data preprocessing...")
        
        # Step 1: Feature engineering
        processed_data = self._feature_engineering(data)
        
        # Step 2: Handle categorical variables
        processed_data = self._encode_categorical_features(processed_data)
        
        # Step 3: Select features
        X, y = self._select_features(processed_data)
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Original class distribution: {np.bincount(y)}")
        
        # Step 5: Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Step 6: Handle imbalanced data
        if apply_smote:
            X_train_balanced, y_train_balanced = self._balance_data(X_train_scaled, y_train)
            print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")
            return X_train_balanced, X_test_scaled, y_train_balanced, y_test, self.feature_columns
        else:
            return X_train_scaled, X_test_scaled, y_train, y_test, self.feature_columns
    
    def _feature_engineering(self, data):
        """Create additional features from existing data"""
        print("Performing feature engineering...")
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Usage intensity features
        df['usage_intensity'] = (df['monthly_minutes'] + df['monthly_data_gb'] * 10) / df['account_age_days']
        df['calls_per_minute'] = df['monthly_calls'] / (df['monthly_minutes'] + 1)  # +1 to avoid division by zero
        
        # Support interaction features
        df['support_interaction_rate'] = (df['support_calls'] + df['support_tickets']) / df['account_age_days']
        df['total_support_issues'] = df['support_calls'] + df['support_tickets']
        
        # Payment behavior features
        df['payment_reliability'] = 1 / (df['late_payments'] + 1)
        df['avg_payment_delay'] = df['payment_delays'] / (df['account_age_days'] / 30)
        
        # Account lifecycle features
        df['account_age_months'] = df['account_age_days'] / 30
        df['is_new_customer'] = (df['account_age_days'] < 90).astype(int)
        df['is_long_term_customer'] = (df['account_age_days'] > 730).astype(int)
        
        # Value-based features
        df['value_per_month'] = df['customer_value'] / (df['account_age_days'] / 30)
        df['is_high_value'] = (df['customer_value'] > df['customer_value'].quantile(0.75)).astype(int)
        
        # Risk indicators
        df['payment_risk_score'] = (df['late_payments'] * 0.4 + df['payment_delays'] * 0.6)
        df['support_risk_score'] = (df['support_calls'] * 0.7 + df['support_tickets'] * 0.3)
        
        # Usage trend features
        df['usage_declining'] = (df['usage_trend'] < -0.3).astype(int)
        df['low_satisfaction'] = (df['satisfaction_score'] < 3).astype(int)
        
        return df
    
    def _encode_categorical_features(self, data):
        """Encode categorical variables"""
        print("Encoding categorical features...")
        
        df = data.copy()
        
        # Encode service plan
        if 'service_plan' in df.columns:
            self.label_encoders['service_plan'] = LabelEncoder()
            df['service_plan_encoded'] = self.label_encoders['service_plan'].fit_transform(df['service_plan'])
        
        # Create dummy variables for contract length
        df['contract_1_month'] = (df['contract_length'] == 1).astype(int)
        df['contract_12_month'] = (df['contract_length'] == 12).astype(int)
        df['contract_24_month'] = (df['contract_length'] == 24).astype(int)
        df['contract_36_month'] = (df['contract_length'] == 36).astype(int)
        
        return df
    
    def _select_features(self, data):
        """Select features for modeling"""
        print("Selecting features for modeling...")
        
        # Define feature columns (excluding target and ID)
        exclude_columns = ['customer_id', 'churned', 'churn_probability', 'service_plan', 'contract_length']
        
        # Get all numeric columns except excluded ones
        feature_columns = [col for col in data.columns 
                          if col not in exclude_columns and data[col].dtype in ['int64', 'float64']]
        
        self.feature_columns = feature_columns
        
        # Prepare feature matrix and target
        X = data[feature_columns].copy()
        y = data['churned'].copy()
        
        # Handle any missing values
        X = X.fillna(X.median())
        
        print(f"Selected {len(feature_columns)} features:")
        for i, col in enumerate(feature_columns, 1):
            print(f"  {i:2d}. {col}")
        
        return X, y
    
    def _balance_data(self, X, y):
        """Apply SMOTE to balance the dataset"""
        print("Applying SMOTE for data balancing...")
        
        try:
            X_balanced, y_balanced = self.smote.fit_resample(X, y)
            return X_balanced, y_balanced
        except Exception as e:
            print(f"SMOTE failed: {e}")
            print("Using original data without balancing...")
            return X, y
    
    def get_feature_importance_names(self):
        """Return feature names for model interpretation"""
        return self.feature_columns if self.feature_columns else []

def main():
    """Test the preprocessor with sample data"""
    from data_generator import generate_customer_data
    
    # Generate sample data
    data = generate_customer_data(n_customers=1000)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Process data
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_data(data)
    
    print(f"\nPreprocessing completed!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")

if __name__ == "__main__":
    main()
