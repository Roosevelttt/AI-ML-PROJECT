"""
Model Training Module for AI/ML Code Classifier

This module handles the training, evaluation, and optimization of machine learning
models for classifying Python code as AI-generated or human-written.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

class CodeClassifierTrainer:
    """Handles training and evaluation of code classification models."""
    
    def __init__(self, data_dir: Path, model_dir: Path):
        """Initialize the model trainer.
        
        Args:
            data_dir: Directory containing training data
            model_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test datasets.
        
        Returns:
            Tuple of (training_data, test_data)
        """
        train_path = self.data_dir / 'training_data.csv'
        test_path = self.data_dir / 'test_data.csv'
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_path}")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"✅ Loaded training data: {len(train_df)} samples")
        print(f"✅ Loaded test data: {len(test_df)} samples")
        
        return train_df, test_df
    
    def prepare_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features for training.
        
        Args:
            train_df: Training dataset
            test_df: Test dataset
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("🔧 Preparing features...")
        
        # Identify feature columns (exclude metadata columns)
        exclude_columns = ['label', 'label_text', 'category', 'complexity', 'code']
        self.feature_columns = [col for col in train_df.columns if col not in exclude_columns]
        
        print(f"📊 Using {len(self.feature_columns)} features")
        
        # Extract features and labels
        X_train = train_df[self.feature_columns]
        X_test = test_df[self.feature_columns]
        y_train = train_df['label']
        y_test = test_df['label']
        
        # Handle categorical features
        categorical_features = X_train.select_dtypes(include=['object']).columns
        if len(categorical_features) > 0:
            print(f"🔤 Encoding {len(categorical_features)} categorical features")
            
            for col in categorical_features:
                # Simple label encoding for categorical features
                combined_values = pd.concat([X_train[col], X_test[col]])
                le = LabelEncoder()
                le.fit(combined_values)
                
                X_train[col] = le.transform(X_train[col])
                X_test[col] = le.transform(X_test[col])
        
        # Convert to numpy arrays
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values
        
        # Scale features
        print("📏 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Feature preparation completed")
        print(f"   Training set: {X_train_scaled.shape}")
        print(f"   Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize different machine learning models.
        
        Returns:
            Dictionary of model instances
        """
        print("🤖 Initializing machine learning models...")
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            ),
            
            'Support Vector Machine': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            
            'Naive Bayes': GaussianNB()
        }
        
        print(f"✅ Initialized {len(models)} models")
        return models
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        print("🚀 Training models...")
        print("=" * 50)
        
        models = self.initialize_models()
        trained_models = {}
        
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                print(f"   ✅ {name} trained successfully")
                print(f"   📊 CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
        
        self.models = trained_models
        print(f"\n✅ Successfully trained {len(trained_models)} models")
        return trained_models
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        print("\n📊 Evaluating models...")
        print("=" * 50)
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            print(f"\n🔍 Evaluating {name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, average='weighted')
                }
                
                if y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                
                evaluation_results[name] = metrics
                
                # Print results
                print(f"   📈 Accuracy:  {metrics['accuracy']:.3f}")
                print(f"   📈 Precision: {metrics['precision']:.3f}")
                print(f"   📈 Recall:    {metrics['recall']:.3f}")
                print(f"   📈 F1-Score:  {metrics['f1_score']:.3f}")
                if 'roc_auc' in metrics:
                    print(f"   📈 ROC-AUC:   {metrics['roc_auc']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error evaluating {name}: {e}")
                evaluation_results[name] = {'error': str(e)}
        
        return evaluation_results
    
    def select_best_model(self, evaluation_results: Dict[str, Dict[str, float]]) -> str:
        """Select the best performing model.
        
        Args:
            evaluation_results: Evaluation results for all models
            
        Returns:
            Name of the best model
        """
        print("\n🏆 Selecting best model...")
        
        best_score = 0
        best_model_name = None
        
        for name, metrics in evaluation_results.items():
            if 'error' not in metrics:
                # Use F1-score as the primary metric
                score = metrics.get('f1_score', 0)
                if score > best_score:
                    best_score = score
                    best_model_name = name
        
        if best_model_name:
            self.best_model = self.models[best_model_name]
            self.best_model_name = best_model_name
            print(f"🥇 Best model: {best_model_name} (F1-Score: {best_score:.3f})")
        else:
            print("❌ No valid model found")
        
        return best_model_name
    
    def save_models(self) -> None:
        """Save trained models and preprocessing components."""
        print("\n💾 Saving models...")
        
        # Save the best model
        if self.best_model:
            model_path = self.model_dir / 'best_model.joblib'
            joblib.dump(self.best_model, model_path)
            print(f"✅ Best model saved to {model_path}")
        
        # Save all models
        for name, model in self.models.items():
            safe_name = name.replace(' ', '_').lower()
            model_path = self.model_dir / f'{safe_name}.joblib'
            joblib.dump(model, model_path)
            print(f"✅ {name} saved to {model_path}")
        
        # Save preprocessing components
        scaler_path = self.model_dir / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved to {scaler_path}")
        
        # Save feature columns
        features_path = self.model_dir / 'feature_columns.joblib'
        joblib.dump(self.feature_columns, features_path)
        print(f"✅ Feature columns saved to {features_path}")
        
        # Save model metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'feature_count': len(self.feature_columns),
            'model_names': list(self.models.keys())
        }
        metadata_path = self.model_dir / 'model_metadata.joblib'
        joblib.dump(metadata, metadata_path)
        print(f"✅ Model metadata saved to {metadata_path}")
    
    def generate_detailed_report(self, evaluation_results: Dict[str, Dict[str, float]], 
                               X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Generate a detailed evaluation report.
        
        Args:
            evaluation_results: Evaluation results for all models
            X_test: Test features
            y_test: Test labels
        """
        print("\n📄 Generating detailed evaluation report...")
        
        report_path = self.model_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("AI/ML Code Classifier - Model Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall summary
            f.write("📊 EVALUATION SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Test samples: {len(y_test)}\n")
            f.write(f"Features used: {len(self.feature_columns)}\n")
            f.write(f"Models trained: {len(self.models)}\n")
            f.write(f"Best model: {self.best_model_name}\n\n")
            
            # Model comparison
            f.write("🏆 MODEL COMPARISON\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
            f.write("-" * 70 + "\n")
            
            for name, metrics in evaluation_results.items():
                if 'error' not in metrics:
                    f.write(f"{name:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
                           f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f}\n")
            
            f.write("\n")
            
            # Detailed results for best model
            if self.best_model:
                f.write(f"🔍 DETAILED RESULTS - {self.best_model_name}\n")
                f.write("-" * 30 + "\n")
                
                y_pred = self.best_model.predict(X_test)
                
                # Classification report
                f.write("Classification Report:\n")
                f.write(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
                f.write("\n")
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                f.write("Confusion Matrix:\n")
                f.write(f"                Predicted\n")
                f.write(f"Actual    Human  AI\n")
                f.write(f"Human     {cm[0,0]:<6} {cm[0,1]:<6}\n")
                f.write(f"AI        {cm[1,0]:<6} {cm[1,1]:<6}\n\n")
            
            # Feature importance (if available)
            if hasattr(self.best_model, 'feature_importances_'):
                f.write("🎯 FEATURE IMPORTANCE (Top 10)\n")
                f.write("-" * 30 + "\n")
                
                importances = self.best_model.feature_importances_
                feature_importance = list(zip(self.feature_columns, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                for i, (feature, importance) in enumerate(feature_importance[:10], 1):
                    f.write(f"{i:2d}. {feature:<30} {importance:.4f}\n")
        
        print(f"📄 Detailed report saved to {report_path}")
    
    def train_and_evaluate(self) -> Dict[str, Any]:
        """Complete training and evaluation pipeline.
        
        Returns:
            Dictionary containing training results
        """
        print("🚀 Starting complete training and evaluation pipeline...")
        print("=" * 70)
        
        try:
            # Load data
            train_df, test_df = self.load_data()
            
            # Prepare features
            X_train, X_test, y_train, y_test = self.prepare_features(train_df, test_df)
            
            # Train models
            trained_models = self.train_models(X_train, y_train)
            
            # Evaluate models
            evaluation_results = self.evaluate_models(X_test, y_test)
            
            # Select best model
            best_model_name = self.select_best_model(evaluation_results)
            
            # Save models
            self.save_models()
            
            # Generate detailed report
            self.generate_detailed_report(evaluation_results, X_test, y_test)
            
            print("\n✅ Training and evaluation completed successfully!")
            print("=" * 70)
            
            return {
                'success': True,
                'best_model': best_model_name,
                'evaluation_results': evaluation_results,
                'feature_count': len(self.feature_columns),
                'test_samples': len(y_test)
            }
            
        except Exception as e:
            print(f"❌ Error in training pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
