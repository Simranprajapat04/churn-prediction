"""
Machine Learning Models for Customer Churn Prediction
Includes Logistic Regression, Decision Tree, and Neural Network models
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ChurnPredictor:
    """Main class for churn prediction models"""
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Logistic Regression model"""
        print("Training Logistic Regression model...")
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000]
        }
        
        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_lr = grid_search.best_estimator_
        
        # Train final model
        best_lr.fit(X_train, y_train)
        
        # Predictions
        y_pred = best_lr.predict(X_test)
        y_pred_proba = best_lr.predict_proba(X_test)[:, 1]
        
        # Evaluate
        performance = self._evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")
        
        self.models['logistic_regression'] = best_lr
        self.model_performance['logistic_regression'] = performance
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"AUC Score: {performance['auc_score']:.4f}")
        
        return best_lr, performance
    
    def train_decision_tree(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Decision Tree model"""
        print("Training Decision Tree model...")
        
        # Hyperparameter tuning
        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        
        dt = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_dt = grid_search.best_estimator_
        
        # Train final model
        best_dt.fit(X_train, y_train)
        
        # Predictions
        y_pred = best_dt.predict(X_test)
        y_pred_proba = best_dt.predict_proba(X_test)[:, 1]
        
        # Evaluate
        performance = self._evaluate_model(y_test, y_pred, y_pred_proba, "Decision Tree")
        
        self.models['decision_tree'] = best_dt
        self.model_performance['decision_tree'] = performance
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"AUC Score: {performance['auc_score']:.4f}")
        
        return best_dt, performance
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Neural Network model"""
        print("Training Neural Network model...")
        
        # Hyperparameter tuning
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [500, 1000]
        }
        
        nn = MLPClassifier(random_state=42)
        grid_search = GridSearchCV(nn, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)  # Reduced CV for speed
        grid_search.fit(X_train, y_train)
        
        best_nn = grid_search.best_estimator_
        
        # Train final model
        best_nn.fit(X_train, y_train)
        
        # Predictions
        y_pred = best_nn.predict(X_test)
        y_pred_proba = best_nn.predict_proba(X_test)[:, 1]
        
        # Evaluate
        performance = self._evaluate_model(y_test, y_pred, y_pred_proba, "Neural Network")
        
        self.models['neural_network'] = best_nn
        self.model_performance['neural_network'] = performance
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"AUC Score: {performance['auc_score']:.4f}")
        
        return best_nn, performance
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """Evaluate model performance"""
        # Calculate metrics
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        
        performance = {
            'model_name': model_name,
            'auc_score': auc_score,
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'confusion_matrix': cm,
            'classification_report': report,
            'fpr': fpr,
            'tpr': tpr
        }
        
        return performance
    
    def compare_models(self):
        """Compare performance of all trained models"""
        if not self.model_performance:
            print("No models trained yet!")
            return
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, performance in self.model_performance.items():
            comparison_data.append({
                'Model': performance['model_name'],
                'AUC Score': performance['auc_score'],
                'Accuracy': performance['accuracy'],
                'Precision': performance['precision'],
                'Recall': performance['recall'],
                'F1 Score': performance['f1_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC Score', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Find best model
        best_model = comparison_df.iloc[0]
        print(f"\nBest Model: {best_model['Model']} (AUC: {best_model['AUC Score']:.4f})")
        
        return comparison_df
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        if not self.model_performance:
            print("No models trained yet!")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, performance in self.model_performance.items():
            plt.plot(performance['fpr'], performance['tpr'], 
                    label=f"{performance['model_name']} (AUC = {performance['auc_score']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        if not self.model_performance:
            print("No models trained yet!")
            return
        
        n_models = len(self.model_performance)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, performance) in enumerate(self.model_performance.items()):
            cm = performance['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{performance["model_name"]}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance(self, feature_names):
        """Get feature importance for tree-based models"""
        importance_data = {}
        
        if 'decision_tree' in self.models:
            dt_importance = self.models['decision_tree'].feature_importances_
            importance_data['decision_tree'] = dt_importance
        
        if 'logistic_regression' in self.models:
            lr_coef = abs(self.models['logistic_regression'].coef_[0])
            importance_data['logistic_regression'] = lr_coef
        
        return importance_data
    
    def save_models(self, filepath_prefix='churn_model'):
        """Save trained models"""
        for model_name, model in self.models.items():
            filename = f"{filepath_prefix}_{model_name}.joblib"
            joblib.dump(model, filename)
            print(f"Model saved: {filename}")
    
    def load_models(self, filepath_prefix='churn_model'):
        """Load trained models"""
        model_files = {
            'logistic_regression': f"{filepath_prefix}_logistic_regression.joblib",
            'decision_tree': f"{filepath_prefix}_decision_tree.joblib",
            'neural_network': f"{filepath_prefix}_neural_network.joblib"
        }
        
        for model_name, filepath in model_files.items():
            try:
                model = joblib.load(filepath)
                self.models[model_name] = model
                print(f"Model loaded: {filepath}")
            except FileNotFoundError:
                print(f"Model file not found: {filepath}")

def main():
    """Test the models with sample data"""
    from data_generator import generate_customer_data
    from data_preprocessor import DataPreprocessor
    
    # Generate and preprocess data
    data = generate_customer_data(n_customers=1000)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_data(data)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Train all models
    predictor.train_logistic_regression(X_train, y_train, X_test, y_test)
    predictor.train_decision_tree(X_train, y_train, X_test, y_test)
    predictor.train_neural_network(X_train, y_train, X_test, y_test)
    
    # Compare models
    predictor.compare_models()
    
    # Plot results
    predictor.plot_roc_curves()
    predictor.plot_confusion_matrices()

if __name__ == "__main__":
    main()
