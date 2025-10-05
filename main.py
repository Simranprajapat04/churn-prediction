"""
Customer Churn Prediction - Main Application
Easy-to-use script for training and evaluating churn prediction models
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_generator import generate_customer_data, save_sample_data
from data_preprocessor import DataPreprocessor
from models import ChurnPredictor

def print_banner():
    """Print application banner"""
    print("="*70)
    print("           CUSTOMER CHURN PREDICTION SYSTEM")
    print("="*70)
    print("Identify customers likely to leave your service")
    print("Using Machine Learning: Logistic Regression, Decision Trees, Neural Networks")
    print("="*70)

def check_data_file():
    """Check if data file exists, create if not"""
    if not os.path.exists('customer_data.csv'):
        print("No existing data found. Generating sample customer data...")
        save_sample_data()
        return True
    else:
        print("Found existing customer data file.")
        return False

def load_data():
    """Load customer data"""
    try:
        data = pd.read_csv('customer_data.csv')
        print(f"Loaded {len(data)} customer records")
        print(f"Churn rate: {data['churned'].mean():.2%}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def display_data_summary(data):
    """Display summary of the loaded data"""
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    
    print(f"Total customers: {len(data):,}")
    print(f"Churned customers: {data['churned'].sum():,} ({data['churned'].mean():.2%})")
    print(f"Retained customers: {(1-data['churned']).sum():,} ({(1-data['churned']).mean():.2%})")
    
    print("\nKey Metrics:")
    print(f"  Average account age: {data['account_age_days'].mean():.0f} days")
    print(f"  Average monthly minutes: {data['monthly_minutes'].mean():.0f}")
    print(f"  Average monthly data: {data['monthly_data_gb'].mean():.1f} GB")
    print(f"  Average satisfaction score: {data['satisfaction_score'].mean():.1f}/5")
    
    print("\nService Plan Distribution:")
    print(data['service_plan'].value_counts())
    
    print("\nChurn by Service Plan:")
    churn_by_plan = data.groupby('service_plan')['churned'].agg(['count', 'sum', 'mean'])
    churn_by_plan.columns = ['Total', 'Churned', 'Churn Rate']
    churn_by_plan['Churn Rate'] = churn_by_plan['Churn Rate'].apply(lambda x: f"{x:.2%}")
    print(churn_by_plan)

def train_models(data, use_smote=True):
    """Train all machine learning models"""
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_data(
        data, apply_smote=use_smote
    )
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Train models
    models_to_train = [
        ('Logistic Regression', predictor.train_logistic_regression),
        ('Decision Tree', predictor.train_decision_tree),
        ('Neural Network', predictor.train_neural_network)
    ]
    
    for model_name, train_func in models_to_train:
        print(f"\n--- Training {model_name} ---")
        try:
            model, performance = train_func(X_train, y_train, X_test, y_test)
            print(f"✓ {model_name} training completed")
        except Exception as e:
            print(f"✗ {model_name} training failed: {e}")
    
    return predictor, feature_names

def display_results(predictor, feature_names):
    """Display model results and insights"""
    print("\n" + "="*50)
    print("MODEL RESULTS")
    print("="*50)
    
    # Compare models
    comparison_df = predictor.compare_models()
    
    # Feature importance
    print("\n" + "="*30)
    print("FEATURE IMPORTANCE")
    print("="*30)
    
    importance_data = predictor.get_feature_importance(feature_names)
    
    for model_name, importance_scores in importance_data.items():
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        # Display top 10 features
        print(feature_importance_df.head(10).to_string(index=False, float_format='%.4f'))
    
    return comparison_df

def save_results(predictor, comparison_df):
    """Save models and results"""
    print("\n" + "="*30)
    print("SAVING RESULTS")
    print("="*30)
    
    # Save models
    predictor.save_models()
    
    # Save comparison results
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("Model comparison saved to 'model_comparison.csv'")
    
    # Generate plots
    try:
        predictor.plot_roc_curves()
        predictor.plot_confusion_matrices()
        print("Plots saved: roc_curves.png, confusion_matrices.png")
    except Exception as e:
        print(f"Error generating plots: {e}")

def predict_churn_for_sample(predictor, data, feature_names, n_samples=5):
    """Predict churn for a sample of customers"""
    print("\n" + "="*30)
    print("SAMPLE PREDICTIONS")
    print("="*30)
    
    # Get a sample of customers
    sample_customers = data.sample(n=n_samples, random_state=42)
    
    # Preprocess the sample (same as training data)
    preprocessor = DataPreprocessor()
    sample_processed = preprocessor._feature_engineering(sample_customers)
    sample_processed = preprocessor._encode_categorical_features(sample_processed)
    
    # Get features for the sample
    sample_features = sample_processed[feature_names].fillna(sample_processed[feature_names].median())
    sample_features_scaled = preprocessor.scaler.transform(sample_features)
    
    # Get predictions from the best model
    best_model_name = list(predictor.models.keys())[0]  # Use first available model
    best_model = predictor.models[best_model_name]
    
    predictions = best_model.predict(sample_features_scaled)
    probabilities = best_model.predict_proba(sample_features_scaled)[:, 1]
    
    # Display results
    results_df = pd.DataFrame({
        'Customer ID': sample_customers['customer_id'],
        'Actual Churn': sample_customers['churned'],
        'Predicted Churn': predictions,
        'Churn Probability': probabilities,
        'Service Plan': sample_customers['service_plan'],
        'Account Age (days)': sample_customers['account_age_days'],
        'Satisfaction Score': sample_customers['satisfaction_score']
    })
    
    print(results_df.to_string(index=False, float_format='%.3f'))
    
    # Show accuracy for this sample
    accuracy = (predictions == sample_customers['churned']).mean()
    print(f"\nSample prediction accuracy: {accuracy:.2%}")

def main():
    """Main application function"""
    print_banner()
    
    # Check and load data
    data_created = check_data_file()
    data = load_data()
    
    if data is None:
        print("Failed to load data. Exiting...")
        return
    
    # Display data summary
    display_data_summary(data)
    
    # Ask user for SMOTE preference
    print("\n" + "="*50)
    print("MODEL TRAINING OPTIONS")
    print("="*50)
    print("SMOTE (Synthetic Minority Oversampling Technique) helps balance the dataset")
    print("by creating synthetic samples for the minority class (churned customers).")
    
    use_smote = True  # Default to using SMOTE
    print(f"Using SMOTE: {use_smote}")
    
    # Train models
    predictor, feature_names = train_models(data, use_smote=use_smote)
    
    # Display results
    comparison_df = display_results(predictor, feature_names)
    
    # Save results
    save_results(predictor, comparison_df)
    
    # Sample predictions
    predict_churn_for_sample(predictor, data, feature_names)
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("Files created:")
    print("  - customer_data.csv (customer dataset)")
    print("  - model_comparison.csv (model performance comparison)")
    print("  - churn_model_*.joblib (trained models)")
    print("  - roc_curves.png (ROC curve comparison)")
    print("  - confusion_matrices.png (confusion matrices)")
    print("\nNext steps:")
    print("  1. Review model performance in model_comparison.csv")
    print("  2. Examine feature importance to understand churn drivers")
    print("  3. Use trained models to predict churn for new customers")
    print("  4. Implement targeted retention strategies based on insights")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your data and try again.")
