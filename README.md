# Customer Churn Prediction System

A comprehensive machine learning solution to identify customers likely to leave a service or subscription. This project implements three different algorithms (Logistic Regression, Decision Trees, and Neural Networks) with advanced data preprocessing including SMOTE for handling imbalanced datasets.

## 🎯 Features

- **Multiple ML Algorithms**: Logistic Regression, Decision Trees, Neural Networks
- **Advanced Data Preprocessing**: SMOTE for handling imbalanced data
- **Feature Engineering**: Creates meaningful features from customer data
- **Model Comparison**: Comprehensive evaluation and comparison of models
- **Visualization**: ROC curves and confusion matrices
- **Easy to Use**: Simple command-line interface
- **Realistic Data**: Synthetic customer data generator for demonstration

## 📊 What is Customer Churn?

Customer churn refers to customers who stop using a company's service or product. Predicting churn is crucial for businesses because:

- **Cost**: Acquiring new customers is 5-25x more expensive than retaining existing ones
- **Revenue Impact**: Even a 5% reduction in churn can increase profits by 25-95%
- **Strategic Planning**: Helps identify at-risk customers for targeted retention campaigns

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project files
# Navigate to the project directory

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Analysis

```bash
python main.py
```

That's it! The system will:
- Generate sample customer data (if not already present)
- Preprocess the data with feature engineering
- Train three machine learning models
- Compare model performance
- Generate visualizations
- Save results and trained models

## 📁 Project Structure

```
├── main.py                 # Main application script
├── data_generator.py       # Synthetic customer data generator
├── data_preprocessor.py    # Data preprocessing and SMOTE
├── models.py              # ML models and evaluation
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── customer_data.csv     # Generated customer dataset
├── model_comparison.csv  # Model performance comparison
├── churn_model_*.joblib  # Trained model files
├── roc_curves.png       # ROC curve visualization
└── confusion_matrices.png # Confusion matrix plots
```

## 📈 Sample Data Features

The system generates realistic customer data including:

- **Account Information**: Age, service plan, contract length
- **Usage Metrics**: Monthly minutes, data usage, call frequency
- **Support Interactions**: Support calls, tickets, satisfaction scores
- **Payment Behavior**: Payment delays, late payments
- **Engagement**: Usage trends, customer value scores

## 🤖 Machine Learning Models

### 1. Logistic Regression
- **Best for**: Interpretable results, baseline performance
- **Pros**: Fast training, easy to interpret, good for binary classification
- **Use case**: When you need to understand which factors drive churn

### 2. Decision Tree
- **Best for**: Non-linear relationships, feature importance
- **Pros**: Handles non-linear patterns, provides feature importance
- **Use case**: When you want to understand decision rules and feature importance

### 3. Neural Network
- **Best for**: Complex patterns, high accuracy
- **Pros**: Can capture complex interactions, often highest accuracy
- **Use case**: When you have large datasets and want maximum predictive power

## 📊 Model Evaluation

The system provides comprehensive evaluation including:

- **AUC Score**: Area under ROC curve (higher is better)
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions
- **ROC Curves**: Visual comparison of model performance

## 🔧 Advanced Usage

### Custom Data
Replace `customer_data.csv` with your own customer data. Ensure it includes:
- Customer ID column
- Feature columns (usage, support, payment data)
- Target column named 'churned' (0/1)

### Model Tuning
Edit the hyperparameter grids in `models.py` to customize model training:
```python
param_grid = {
    'C': [0.1, 1, 10, 100],  # Add more values
    'penalty': ['l1', 'l2'],
    # ... other parameters
}
```

### Feature Engineering
Modify `_feature_engineering()` in `data_preprocessor.py` to create domain-specific features:
```python
def _feature_engineering(self, data):
    # Add your custom features here
    df['custom_feature'] = df['feature1'] * df['feature2']
    return df
```

## 📋 Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- plotly
- joblib

## 🎯 Business Applications

### 1. Customer Retention Campaigns
- Identify high-risk customers for targeted retention efforts
- Prioritize customers by churn probability and value
- Develop personalized retention strategies

### 2. Resource Allocation
- Focus support resources on at-risk customers
- Optimize customer success team efforts
- Plan capacity based on predicted churn

### 3. Product Development
- Identify features that reduce churn
- Understand customer pain points
- Guide product roadmap decisions

### 4. Pricing Strategy
- Understand price sensitivity and churn correlation
- Test retention offers for high-risk segments
- Optimize pricing for customer retention

## 📊 Sample Results

Typical performance metrics:
- **Logistic Regression**: AUC ~0.75-0.85
- **Decision Tree**: AUC ~0.80-0.90
- **Neural Network**: AUC ~0.85-0.95

## 🔍 Understanding the Results

### ROC Curve
- **X-axis**: False Positive Rate (1 - Specificity)
- **Y-axis**: True Positive Rate (Sensitivity)
- **Closer to top-left corner**: Better model
- **Diagonal line**: Random classifier baseline

### Confusion Matrix
```
                Predicted
Actual     No Churn    Churn
No Churn   True Neg    False Pos
Churn      False Neg   True Pos
```

### Feature Importance
Shows which customer attributes are most predictive of churn:
- High support interactions
- Payment delays
- Low satisfaction scores
- Declining usage patterns

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Error**: Reduce dataset size in `data_generator.py`
2. **Slow Training**: Reduce hyperparameter grid size in `models.py`
3. **Import Errors**: Ensure all packages are installed: `pip install -r requirements.txt`
4. **Plot Display Issues**: Add `plt.show()` if plots don't display

### Performance Tips

- Use `n_jobs=-1` for parallel processing
- Reduce CV folds for faster training
- Sample data for initial testing
- Use simpler models for large datasets

## 📚 Further Reading

- [SMOTE Paper](https://arxiv.org/abs/1106.1813) - Synthetic Minority Oversampling Technique
- [Customer Churn Analysis](https://en.wikipedia.org/wiki/Customer_attrition) - Wikipedia
- [Scikit-learn Documentation](https://scikit-learn.org/stable/) - ML library documentation

## 🤝 Contributing

Feel free to contribute improvements:
- Add new ML algorithms
- Improve feature engineering
- Enhance visualizations
- Add more evaluation metrics

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Churn Prediction! 🎯**

For questions or issues, please check the code comments or create an issue in the repository.
