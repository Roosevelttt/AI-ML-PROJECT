"""
Model Training Script for Python Code Classifier

This script trains machine learning models to classify Python code as
AI-generated or human-written using the prepared dataset.
"""

import sys
import argparse
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.model_trainer import CodeClassifierTrainer
from config import Config

def main():
    """Main function to run model training."""
    parser = argparse.ArgumentParser(description='Train Python Code Classifier models')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick training with fewer models')
    
    args = parser.parse_args()
    
    print("🚀 Starting Python Code Classifier Model Training")
    print("=" * 60)
    
    try:
        # Initialize configuration
        config = Config()
        
        # Check if dataset exists
        train_path = config.DATA_DIR / 'training_data.csv'
        if not train_path.exists():
            print("❌ Training dataset not found!")
            print("Please run 'python create_dataset.py' first to create the dataset.")
            return False
        
        # Initialize trainer
        trainer = CodeClassifierTrainer(config.DATA_DIR, config.MODEL_DIR)
        
        # Run training and evaluation
        results = trainer.train_and_evaluate()
        
        if results['success']:
            print("\n🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"🏆 Best Model: {results['best_model']}")
            print(f"📊 Features Used: {results['feature_count']}")
            print(f"🧪 Test Samples: {results['test_samples']}")
            
            print("\n📈 Model Performance Summary:")
            print("-" * 40)
            for model_name, metrics in results['evaluation_results'].items():
                if 'error' not in metrics:
                    print(f"{model_name}:")
                    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
                    print(f"  F1-Score:  {metrics['f1_score']:.3f}")
                    if 'roc_auc' in metrics:
                        print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
                    print()
            
            print("📁 Saved Files:")
            print(f"  - Best model: {config.MODEL_DIR}/best_model.joblib")
            print(f"  - Scaler: {config.MODEL_DIR}/scaler.joblib")
            print(f"  - Features: {config.MODEL_DIR}/feature_columns.joblib")
            print(f"  - Report: {config.MODEL_DIR}/evaluation_report.txt")
            
            print("\n✅ Models are ready for use in the web application!")
            
            if args.verbose:
                print("\n🔍 DETAILED RESULTS:")
                print("-" * 40)
                for model_name, metrics in results['evaluation_results'].items():
                    if 'error' not in metrics:
                        print(f"\n{model_name} Detailed Metrics:")
                        for metric, value in metrics.items():
                            print(f"  {metric}: {value:.4f}")
            
            return True
            
        else:
            print(f"❌ Training failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
