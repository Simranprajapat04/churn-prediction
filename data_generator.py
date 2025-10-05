"""
Customer Data Generator
Creates synthetic customer data for churn prediction demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_customer_data(n_customers=1000, churn_rate=0.2):
    """
    Generate synthetic customer data with realistic patterns
    
    Args:
        n_customers (int): Number of customers to generate
        churn_rate (float): Proportion of customers who churn
    
    Returns:
        pd.DataFrame: Generated customer data
    """
    np.random.seed(42)
    random.seed(42)
    
    # Customer IDs
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
    
    # Account age (days since registration)
    account_age_days = np.random.exponential(scale=365, size=n_customers).astype(int)
    account_age_days = np.clip(account_age_days, 1, 1825)  # 1 day to 5 years
    
    # Monthly usage metrics
    monthly_minutes = np.random.lognormal(mean=5.5, sigma=0.8, size=n_customers)
    monthly_minutes = np.clip(monthly_minutes, 0, 2000)
    
    monthly_data_gb = np.random.lognormal(mean=3.5, sigma=1.0, size=n_customers)
    monthly_data_gb = np.clip(monthly_data_gb, 0, 100)
    
    monthly_calls = np.random.poisson(lam=15, size=n_customers)
    monthly_calls = np.clip(monthly_calls, 0, 100)
    
    # Support interactions
    support_calls = np.random.poisson(lam=2, size=n_customers)
    support_calls = np.clip(support_calls, 0, 20)
    
    support_tickets = np.random.poisson(lam=1.5, size=n_customers)
    support_tickets = np.clip(support_tickets, 0, 15)
    
    # Payment behavior
    payment_delays = np.random.exponential(scale=5, size=n_customers).astype(int)
    payment_delays = np.clip(payment_delays, 0, 30)
    
    late_payments = np.random.binomial(n=12, p=0.1, size=n_customers)  # 12 months, 10% chance per month
    late_payments = np.clip(late_payments, 0, 12)
    
    # Service plan
    plan_types = ['Basic', 'Premium', 'Enterprise']
    plan_weights = [0.5, 0.35, 0.15]
    service_plan = np.random.choice(plan_types, size=n_customers, p=plan_weights)
    
    # Contract length
    contract_length = np.random.choice([1, 12, 24, 36], size=n_customers, p=[0.1, 0.4, 0.3, 0.2])
    
    # Customer satisfaction (1-5 scale)
    satisfaction_score = np.random.beta(a=2, b=2, size=n_customers) * 4 + 1
    satisfaction_score = np.clip(satisfaction_score, 1, 5)
    
    # Feature engineering for churn prediction
    # Usage trend (declining usage indicates potential churn)
    usage_trend = np.random.normal(0, 0.3, size=n_customers)
    
    # Customer value score
    customer_value = (monthly_minutes * 0.1 + monthly_data_gb * 0.5 + 
                     monthly_calls * 0.2 + (satisfaction_score - 1) * 10)
    
    # Create churn labels based on multiple factors
    churn_probability = (
        0.3 * (support_calls > 5) +  # High support calls
        0.4 * (late_payments > 3) +  # Multiple late payments
        0.3 * (satisfaction_score < 2.5) +  # Low satisfaction
        0.2 * (usage_trend < -0.5) +  # Declining usage
        0.1 * (account_age_days > 1000) +  # Old account (potential lifecycle end)
        0.2 * (payment_delays > 15)  # Recent payment delays
    )
    
    # Add some randomness
    churn_probability += np.random.normal(0, 0.1, size=n_customers)
    churn_probability = np.clip(churn_probability, 0, 1)
    
    # Generate actual churn based on probability
    churned = (churn_probability > (1 - churn_rate)).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'customer_id': customer_ids,
        'account_age_days': account_age_days,
        'monthly_minutes': monthly_minutes,
        'monthly_data_gb': monthly_data_gb,
        'monthly_calls': monthly_calls,
        'support_calls': support_calls,
        'support_tickets': support_tickets,
        'payment_delays': payment_delays,
        'late_payments': late_payments,
        'service_plan': service_plan,
        'contract_length': contract_length,
        'satisfaction_score': satisfaction_score,
        'usage_trend': usage_trend,
        'customer_value': customer_value,
        'churn_probability': churn_probability,
        'churned': churned
    })
    
    return data

def save_sample_data():
    """Generate and save sample data"""
    print("Generating customer data...")
    data = generate_customer_data(n_customers=2000, churn_rate=0.25)
    
    # Save to CSV
    data.to_csv('customer_data.csv', index=False)
    print(f"Generated {len(data)} customer records")
    print(f"Churn rate: {data['churned'].mean():.2%}")
    print(f"Data saved to 'customer_data.csv'")
    
    return data

if __name__ == "__main__":
    data = save_sample_data()
    print("\nSample data:")
    print(data.head())
    print("\nData info:")
    print(data.info())
    print("\nChurn distribution:")
    print(data['churned'].value_counts())
