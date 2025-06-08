"""
Model Prediction Module for AI/ML Code Classifier

This module handles loading trained models and making predictions on new code samples.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from feature_extractor import CodeFeatureExtractor

class CodeClassifierPredictor:
    """Handles loading trained models and making predictions."""
    
    def __init__(self, model_dir: Path):
        """Initialize the predictor.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.feature_extractor = CodeFeatureExtractor()
        self.metadata = None
        
        # Load model components
        self._load_model_components()
    
    def _load_model_components(self) -> None:
        """Load all necessary model components."""
        try:
            # Load best model
            model_path = self.model_dir / 'best_model.joblib'
            if model_path.exists():
                self.model = joblib.load(model_path)
                print(f"✅ Loaded best model from {model_path}")
            else:
                raise FileNotFoundError(f"Best model not found at {model_path}")
            
            # Load scaler
            scaler_path = self.model_dir / 'scaler.joblib'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Loaded scaler from {scaler_path}")
            else:
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            
            # Load feature columns
            features_path = self.model_dir / 'feature_columns.joblib'
            if features_path.exists():
                self.feature_columns = joblib.load(features_path)
                print(f"✅ Loaded feature columns ({len(self.feature_columns)} features)")
            else:
                raise FileNotFoundError(f"Feature columns not found at {features_path}")
            
            # Load metadata
            metadata_path = self.model_dir / 'model_metadata.joblib'
            if metadata_path.exists():
                self.metadata = joblib.load(metadata_path)
                print(f"✅ Loaded model metadata")
                print(f"   Best model: {self.metadata.get('best_model_name', 'Unknown')}")
            else:
                print("⚠️  Model metadata not found")
                
        except Exception as e:
            print(f"❌ Error loading model components: {e}")
            raise
    
    def extract_and_prepare_features(self, code: str) -> np.ndarray:
        """Extract features from code and prepare for prediction.
        
        Args:
            code: Python code as string
            
        Returns:
            Prepared feature array
        """
        # Extract features
        features = self.feature_extractor.extract_features(code)
        
        # Create DataFrame with all required columns
        feature_df = pd.DataFrame([features])
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in feature_df.columns:
                # Add missing columns with default values
                if col in ['quote_preference', 'indentation_preference', 'naming_preference']:
                    feature_df[col] = 'unknown'
                elif col == 'has_main_guard':
                    feature_df[col] = False
                else:
                    feature_df[col] = 0
        
        # Select only the required columns in the correct order
        feature_df = feature_df[self.feature_columns]
        
        # Handle categorical features (simple label encoding)
        categorical_columns = feature_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col == 'quote_preference':
                mapping = {'single': 0, 'double': 1, 'none': 2, 'unknown': 3}
            elif col == 'indentation_preference':
                mapping = {'spaces': 0, 'tabs': 1, 'none': 2, 'unknown': 3}
            elif col == 'naming_preference':
                mapping = {'snake_case': 0, 'camel_case': 1, 'none': 2, 'unknown': 3}
            else:
                mapping = {'unknown': 0}
            
            feature_df[col] = feature_df[col].map(mapping).fillna(0)
        
        # Convert to numpy array
        feature_array = feature_df.values
        
        # Scale features
        feature_array_scaled = self.scaler.transform(feature_array)
        
        return feature_array_scaled
    
    def predict(self, code: str) -> Dict[str, Any]:
        """Make a prediction on a code sample.
        
        Args:
            code: Python code as string
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            if not self.model:
                raise ValueError("Model not loaded")
            
            # Extract and prepare features
            features = self.extract_and_prepare_features(code)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            prediction_proba = self.model.predict_proba(features)[0] if hasattr(self.model, 'predict_proba') else None
            
            # Prepare result
            result = {
                'prediction': int(prediction),
                'prediction_text': 'ai-generated' if prediction == 1 else 'human-written',
                'confidence': float(max(prediction_proba)) if prediction_proba is not None else 0.5,
                'probabilities': {
                    'human-written': float(prediction_proba[0]) if prediction_proba is not None else 0.5,
                    'ai-generated': float(prediction_proba[1]) if prediction_proba is not None else 0.5
                },
                'model_name': self.metadata.get('best_model_name', 'Unknown') if self.metadata else 'Unknown'
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return {
                'error': str(e),
                'prediction': None,
                'prediction_text': None,
                'confidence': 0.0
            }
    
    def predict_batch(self, code_samples: List[str]) -> List[Dict[str, Any]]:
        """Make predictions on multiple code samples.
        
        Args:
            code_samples: List of Python code strings
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"🔍 Making predictions on {len(code_samples)} code samples...")
        
        for i, code in enumerate(code_samples):
            if i % 10 == 0:
                print(f"   Processing sample {i+1}/{len(code_samples)}")
            
            result = self.predict(code)
            results.append(result)
        
        print(f"✅ Completed predictions on {len(code_samples)} samples")
        return results
    
    def analyze_code_features(self, code: str) -> Dict[str, Any]:
        """Analyze code features in detail.
        
        Args:
            code: Python code as string
            
        Returns:
            Dictionary containing detailed feature analysis
        """
        try:
            # Extract raw features
            raw_features = self.feature_extractor.extract_features(code)
            
            # Make prediction
            prediction_result = self.predict(code)
            
            # Categorize features
            basic_features = {
                'total_characters': raw_features.get('total_characters', 0),
                'total_lines': raw_features.get('total_lines', 0),
                'non_empty_lines': raw_features.get('non_empty_lines', 0),
                'avg_line_length': raw_features.get('avg_line_length', 0),
                'whitespace_ratio': raw_features.get('whitespace_ratio', 0)
            }
            
            syntactic_features = {
                'num_functions': raw_features.get('num_functions', 0),
                'num_classes': raw_features.get('num_classes', 0),
                'num_imports': raw_features.get('num_imports', 0),
                'num_variables': raw_features.get('num_variables', 0),
                'cyclomatic_complexity': raw_features.get('cyclomatic_complexity', 0)
            }
            
            style_features = {
                'quote_preference': raw_features.get('quote_preference', 'unknown'),
                'indentation_preference': raw_features.get('indentation_preference', 'unknown'),
                'naming_preference': raw_features.get('naming_preference', 'unknown'),
                'comment_ratio': raw_features.get('comment_ratio', 0)
            }
            
            documentation_features = {
                'num_docstrings': raw_features.get('num_docstrings', 0),
                'type_hints_count': raw_features.get('type_hints_count', 0),
                'return_annotations_count': raw_features.get('return_annotations_count', 0),
                'has_main_guard': raw_features.get('has_main_guard', False)
            }
            
            complexity_features = {
                'flesch_reading_ease': raw_features.get('flesch_reading_ease', 0),
                'token_diversity': raw_features.get('token_diversity', 0),
                'unique_tokens': raw_features.get('unique_tokens', 0),
                'total_tokens': raw_features.get('total_tokens', 0)
            }
            
            return {
                'prediction': prediction_result,
                'feature_analysis': {
                    'basic': basic_features,
                    'syntactic': syntactic_features,
                    'style': style_features,
                    'documentation': documentation_features,
                    'complexity': complexity_features
                },
                'raw_features': raw_features
            }
            
        except Exception as e:
            print(f"❌ Error analyzing code features: {e}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.metadata:
            return {'error': 'Model metadata not available'}
        
        return {
            'model_name': self.metadata.get('best_model_name', 'Unknown'),
            'feature_count': self.metadata.get('feature_count', 0),
            'available_models': self.metadata.get('model_names', []),
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'feature_columns_loaded': self.feature_columns is not None
        }
    
    def validate_prediction_setup(self) -> Dict[str, bool]:
        """Validate that all components are properly loaded.
        
        Returns:
            Dictionary indicating which components are ready
        """
        return {
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'feature_columns_loaded': self.feature_columns is not None,
            'feature_extractor_ready': self.feature_extractor is not None,
            'ready_for_prediction': all([
                self.model is not None,
                self.scaler is not None,
                self.feature_columns is not None,
                self.feature_extractor is not None
            ])
        }
