"""
Predictive modeling module for crime type classification.
Implements Random Forest and XGBoost models for crime prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, ensure_dir_exists, save_model, load_model
import logging

logger = logging.getLogger(__name__)

class CrimePredictiveModel:
    """
    Crime type prediction using machine learning models.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize predictive model with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = load_config(config_path) if os.path.exists(config_path) else {}
        self.df = None
        self.models = {}
        self.feature_columns = []
        self.target_encoder = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load processed crime data.
        
        Args:
            file_path (str): Path to processed data file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            self.df = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def prepare_features_target(self, df: pd.DataFrame = None, target_column: str = "crime_type") -> tuple:
        """
        Prepare features and target for machine learning.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target column name
            
        Returns:
            tuple: (X, y) features and target
        """
        if df is None:
            df = self.df
        
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found")
            return None, None
        
        # Select feature columns (exclude target and non-predictive columns)
        exclude_columns = [
            target_column, 'incident_id', 'date', 'state', 'district',
            'month_name', 'weekday_name', 'season'  # Keep encoded versions
        ]
        
        # Select numerical and encoded categorical columns
        feature_columns = [col for col in df.columns 
                          if col not in exclude_columns and 
                          (df[col].dtype in ['int64', 'float64'] or col.endswith('_encoded'))]
        
        # Remove columns with too many missing values or zero variance
        feature_columns = [col for col in feature_columns 
                          if df[col].isnull().sum() / len(df) < 0.5 and 
                          df[col].var() > 0]
        
        self.feature_columns = feature_columns
        
        X = df[feature_columns].fillna(0)
        y = df[target_column]
        
        # Encode target if it's categorical
        if y.dtype == 'object':
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)
        
        logger.info(f"Prepared {len(feature_columns)} features for prediction")
        logger.info(f"Target classes: {len(np.unique(y))}")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   random_state: int = 42) -> tuple:
        """
        Split data into train and test sets.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
            test_size (float): Test set proportion
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train: np.ndarray = None, y_train: np.ndarray = None,
                           **kwargs) -> RandomForestClassifier:
        """
        Train Random Forest classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            RandomForestClassifier: Trained model
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
            
        # Default parameters from config or use defaults
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Override with config if available
        if self.config.get('models', {}).get('classification', {}).get('random_forest'):
            default_params.update(self.config['models']['classification']['random_forest'])
        
        # Override with provided kwargs
        default_params.update(kwargs)
        
        rf_model = RandomForestClassifier(**default_params)
        rf_model.fit(X_train, y_train)
        
        self.models['random_forest'] = rf_model
        
        logger.info("Random Forest model trained successfully")
        return rf_model
    
    def train_xgboost(self, X_train: np.ndarray = None, y_train: np.ndarray = None,
                     **kwargs) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            **kwargs: Additional parameters for XGBClassifier
            
        Returns:
            xgb.XGBClassifier: Trained model
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
            
        # Default parameters from config or use defaults
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        
        # Override with config if available
        if self.config.get('models', {}).get('classification', {}).get('xgboost'):
            default_params.update(self.config['models']['classification']['xgboost'])
        
        # Override with provided kwargs
        default_params.update(kwargs)
        
        xgb_model = xgb.XGBClassifier(**default_params)
        xgb_model.fit(X_train, y_train)
        
        self.models['xgboost'] = xgb_model
        
        logger.info("XGBoost model trained successfully")
        return xgb_model
    
    def evaluate_model(self, model, X_test: np.ndarray = None, y_test: np.ndarray = None,
                      model_name: str = "Model") -> dict:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
            
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted')
        }
        
        logger.info(f"{model_name} Evaluation:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  F1-Score (Macro): {metrics['f1_macro']:.3f}")
        logger.info(f"  F1-Score (Weighted): {metrics['f1_weighted']:.3f}")
        
        return metrics
    
    def plot_feature_importance(self, model, model_name: str, top_n: int = 20, 
                               save_path: str = None) -> None:
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model
            model_name (str): Name of the model
            top_n (int): Number of top features to show
            save_path (str): Path to save the plot
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return
        
        # Get feature importance
        importance = model.feature_importances_
        feature_names = self.feature_columns
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True).tail(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, model, X_test: np.ndarray = None, y_test: np.ndarray = None,
                             model_name: str = "Model", save_path: str = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
            
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Get class labels
        if self.target_encoder:
            class_labels = self.target_encoder.classes_
        else:
            class_labels = [f'Class_{i}' for i in range(len(np.unique(y_test)))]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, metrics_dict: dict, save_path: str = None) -> None:
        """
        Plot comparison of multiple models.
        
        Args:
            metrics_dict (dict): Dictionary of model metrics
            save_path (str): Path to save the plot
        """
        if not metrics_dict:
            return
            
        # Prepare data for plotting
        models = list(metrics_dict.keys())
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [metrics_dict[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def cross_validate_models(self, X: np.ndarray = None, y: np.ndarray = None, 
                             cv: int = 5) -> dict:
        """
        Perform cross-validation for all trained models.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation scores
        """
        if X is None:
            X = np.vstack([self.X_train, self.X_test])
        if y is None:
            y = np.hstack([self.y_train, self.y_test])
            
        cv_results = {}
        
        for model_name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
            cv_results[model_name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            
            logger.info(f"{model_name} CV F1-Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        return cv_results
    
    def hyperparameter_tuning(self, model_type: str, X_train: np.ndarray = None, 
                             y_train: np.ndarray = None) -> dict:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_type (str): Type of model ('random_forest' or 'xgboost')
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            
        Returns:
            dict: Best parameters and score
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
            
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:
            logger.error(f"Unknown model type: {model_type}")
            return {}
        
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro', 
                                  n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
        
        logger.info(f"{model_type} Best parameters: {results['best_params']}")
        logger.info(f"{model_type} Best CV score: {results['best_score']:.3f}")
        
        return results
    
    def predict_crime_type(self, model, features: dict, model_name: str = "Model") -> dict:
        """
        Predict crime type for new data.
        
        Args:
            model: Trained model
            features (dict): Feature values
            model_name (str): Name of the model
            
        Returns:
            dict: Prediction results
        """
        # Prepare feature vector
        feature_vector = np.zeros(len(self.feature_columns))
        
        for i, feature in enumerate(self.feature_columns):
            if feature in features:
                feature_vector[i] = features[feature]
        
        # Make prediction
        prediction = model.predict([feature_vector])[0]
        prediction_proba = model.predict_proba([feature_vector])[0] if hasattr(model, 'predict_proba') else None
        
        # Convert prediction back to original label
        if self.target_encoder:
            predicted_crime = self.target_encoder.inverse_transform([prediction])[0]
        else:
            predicted_crime = prediction
        
        results = {
            'predicted_crime_type': predicted_crime,
            'confidence': max(prediction_proba) if prediction_proba is not None else None,
            'probabilities': dict(zip(self.target_encoder.classes_, prediction_proba)) if self.target_encoder and prediction_proba is not None else None
        }
        
        logger.info(f"{model_name} Prediction: {predicted_crime}")
        if results['confidence']:
            logger.info(f"Confidence: {results['confidence']:.3f}")
        
        return results
    
    def save_models(self, output_dir: str = "models/saved_models") -> None:
        """
        Save trained models and encoders.
        
        Args:
            output_dir (str): Directory to save models
        """
        ensure_dir_exists(output_dir)
        
        for model_name, model in self.models.items():
            model_path = f"{output_dir}/crime_prediction_{model_name}_model.pkl"
            save_model(model, model_path)
        
        if self.target_encoder:
            encoder_path = f"{output_dir}/crime_prediction_target_encoder.pkl"
            save_model(self.target_encoder, encoder_path)
        
        # Save feature columns
        import json
        features_path = f"{output_dir}/crime_prediction_features.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f)
        
        logger.info(f"Predictive models saved to {output_dir}")
    
    def generate_prediction_report(self, df: pd.DataFrame = None, 
                                  output_dir: str = "reports/predictions") -> dict:
        """
        Generate comprehensive prediction model report.
        
        Args:
            df (pd.DataFrame): Input dataframe
            output_dir (str): Directory to save reports
            
        Returns:
            dict: Comprehensive prediction analysis
        """
        if df is None:
            df = self.df
            
        ensure_dir_exists(output_dir)
        
        # Prepare data
        X, y = self.prepare_features_target(df)
        if X is None or y is None:
            logger.error("Failed to prepare features and target")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Train models
        rf_model = self.train_random_forest(X_train, y_train)
        xgb_model = self.train_xgboost(X_train, y_train)
        
        # Evaluate models
        rf_metrics = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        xgb_metrics = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        
        # Generate visualizations
        self.plot_feature_importance(rf_model, "Random Forest", 
                                    save_path=f"{output_dir}/rf_feature_importance.png")
        self.plot_feature_importance(xgb_model, "XGBoost", 
                                    save_path=f"{output_dir}/xgb_feature_importance.png")
        
        self.plot_confusion_matrix(rf_model, X_test, y_test, "Random Forest",
                                 save_path=f"{output_dir}/rf_confusion_matrix.png")
        self.plot_confusion_matrix(xgb_model, X_test, y_test, "XGBoost",
                                 save_path=f"{output_dir}/xgb_confusion_matrix.png")
        
        metrics_dict = {'Random Forest': rf_metrics, 'XGBoost': xgb_metrics}
        self.plot_model_comparison(metrics_dict, 
                                 save_path=f"{output_dir}/model_comparison.png")
        
        # Cross-validation
        cv_results = self.cross_validate_models(X, y)
        
        # Save models
        self.save_models()
        
        # Compile report
        report = {
            'data_info': {
                'total_samples': len(df),
                'features_used': len(self.feature_columns),
                'target_classes': len(np.unique(y)),
                'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
            },
            'model_performance': {
                'random_forest': rf_metrics,
                'xgboost': xgb_metrics
            },
            'cross_validation': cv_results,
            'feature_importance': {
                'random_forest': dict(zip(self.feature_columns, rf_model.feature_importances_)),
                'xgboost': dict(zip(self.feature_columns, xgb_model.feature_importances_))
            },
            'recommendations': self._generate_prediction_recommendations(rf_metrics, xgb_metrics, cv_results)
        }
        
        logger.info(f"Comprehensive prediction report generated in {output_dir}")
        return report
    
    def _generate_prediction_recommendations(self, rf_metrics: dict, xgb_metrics: dict, 
                                           cv_results: dict) -> list:
        """
        Generate recommendations based on model performance.
        
        Args:
            rf_metrics (dict): Random Forest metrics
            xgb_metrics (dict): XGBoost metrics
            cv_results (dict): Cross-validation results
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        # Compare model performance
        if rf_metrics['f1_weighted'] > xgb_metrics['f1_weighted']:
            recommendations.append(
                f"Random Forest performs better (F1: {rf_metrics['f1_weighted']:.3f}) than XGBoost "
                f"(F1: {xgb_metrics['f1_weighted']:.3f}). Consider using Random Forest for deployment."
            )
        else:
            recommendations.append(
                f"XGBoost performs better (F1: {xgb_metrics['f1_weighted']:.3f}) than Random Forest "
                f"(F1: {rf_metrics['f1_weighted']:.3f}). Consider using XGBoost for deployment."
            )
        
        # Accuracy assessment
        best_accuracy = max(rf_metrics['accuracy'], xgb_metrics['accuracy'])
        if best_accuracy < 0.8:
            recommendations.append(
                f"Model accuracy ({best_accuracy:.3f}) could be improved. Consider feature engineering, "
                "hyperparameter tuning, or collecting more data."
            )
        
        # Cross-validation stability
        for model_name, cv_result in cv_results.items():
            if cv_result['std_score'] > 0.1:
                recommendations.append(
                    f"{model_name} shows high variance in cross-validation "
                    f"(std: {cv_result['std_score']:.3f}). Consider regularization or more stable features."
                )
        
        # General recommendations
        recommendations.extend([
            "Monitor model performance regularly and retrain with new data.",
            "Consider ensemble methods combining both models for better predictions.",
            "Implement feature selection to reduce model complexity and improve interpretability.",
            "Use model explanability tools (SHAP, LIME) to understand prediction reasoning."
        ])
        
        return recommendations

def main():
    """
    Main function to demonstrate predictive modeling capabilities.
    """
    # Initialize model
    predictor = CrimePredictiveModel()
    
    # Load processed data
    data_path = "data/processed/crime_data_processed.csv"
    if os.path.exists(data_path):
        df = predictor.load_data(data_path)
    else:
        # Create sample data if processed data doesn't exist
        from src.data_preprocessing import CrimeDataPreprocessor
        from src.utils import create_sample_crime_data, create_sample_demographic_data
        
        crime_data = create_sample_crime_data(1000)
        demographic_data = create_sample_demographic_data()
        
        preprocessor = CrimeDataPreprocessor()
        df = preprocessor.preprocess_pipeline(crime_data, demographic_data)
    
    # Generate comprehensive prediction report
    report = predictor.generate_prediction_report(df)
    
    print("\n" + "="*50)
    print("CRIME PREDICTION MODEL REPORT")
    print("="*50)
    
    print(f"\nData Information:")
    data_info = report.get('data_info', {})
    print(f"  Total Samples: {data_info.get('total_samples', 'N/A'):,}")
    print(f"  Features Used: {data_info.get('features_used', 'N/A')}")
    print(f"  Target Classes: {data_info.get('target_classes', 'N/A')}")
    
    print(f"\nModel Performance:")
    rf_perf = report.get('model_performance', {}).get('random_forest', {})
    xgb_perf = report.get('model_performance', {}).get('xgboost', {})
    
    print(f"  Random Forest:")
    print(f"    Accuracy: {rf_perf.get('accuracy', 0):.3f}")
    print(f"    F1-Score (Weighted): {rf_perf.get('f1_weighted', 0):.3f}")
    
    print(f"  XGBoost:")
    print(f"    Accuracy: {xgb_perf.get('accuracy', 0):.3f}")
    print(f"    F1-Score (Weighted): {xgb_perf.get('f1_weighted', 0):.3f}")
    
    print(f"\nCross-Validation Results:")
    cv_results = report.get('cross_validation', {})
    for model_name, cv_result in cv_results.items():
        print(f"  {model_name}: {cv_result.get('mean_score', 0):.3f} (+/- {cv_result.get('std_score', 0) * 2:.3f})")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(report.get('recommendations', []), 1):
        print(f"  {i}. {rec}")
    
    print("\nPrediction model visualizations saved to 'reports/predictions/' directory")
    
    # Example prediction
    print("\n" + "="*30)
    print("EXAMPLE PREDICTION")
    print("="*30)
    
    if 'random_forest' in predictor.models:
        example_features = {
            'latitude': 28.6139,  # Delhi coordinates
            'longitude': 77.2090,
            'month': 6,  # June
            'weekday': 1,  # Tuesday
            'population_normalized': 0.5,
            'literacy_rate_normalized': 0.7
        }
        
        result = predictor.predict_crime_type(predictor.models['random_forest'], 
                                            example_features, "Random Forest")
        
        print(f"Input Location: Delhi (28.6139, 77.2090)")
        print(f"Predicted Crime Type: {result.get('predicted_crime_type', 'Unknown')}")
        if result.get('confidence'):
            print(f"Confidence: {result['confidence']:.3f}")

if __name__ == "__main__":
    main()