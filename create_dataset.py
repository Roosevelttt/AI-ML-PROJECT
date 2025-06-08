"""
Dataset Creation Script for Python Code Classifier

This script creates the training dataset by collecting code samples,
extracting features, and preparing the data for machine learning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.data_collector import DataCollector
from utils.feature_extractor import CodeFeatureExtractor
from config import Config

def create_training_dataset():
    """Create the complete training dataset."""
    print("🚀 Starting dataset creation process...")
    print("=" * 60)
    
    # Initialize components
    config = Config()
    data_collector = DataCollector(config.DATA_DIR)
    feature_extractor = CodeFeatureExtractor()
    
    # Step 1: Collect code samples
    print("\n📝 Step 1: Collecting code samples")
    print("-" * 40)
    
    human_samples = data_collector.create_human_written_samples()
    ai_samples = data_collector.create_ai_generated_samples()
    
    # Step 2: Combine datasets
    print("\n🔄 Step 2: Combining datasets")
    print("-" * 40)
    
    combined_df = data_collector.combine_datasets(human_samples, ai_samples)
    
    # Step 3: Extract features
    print("\n🔍 Step 3: Extracting features")
    print("-" * 40)
    
    features_df = feature_extractor.extract_features_batch(combined_df['code'].tolist())
    
    # Step 4: Combine features with labels
    print("\n🔗 Step 4: Combining features with labels")
    print("-" * 40)
    
    # Add labels and metadata to features
    final_df = features_df.copy()
    final_df['label'] = combined_df['label']
    final_df['label_text'] = combined_df['label_text']
    final_df['category'] = combined_df['category']
    final_df['complexity'] = combined_df['complexity']
    final_df['code'] = combined_df['code']  # Keep original code for reference
    
    # Step 5: Save datasets
    print("\n💾 Step 5: Saving datasets")
    print("-" * 40)
    
    # Save full dataset
    full_dataset_path = data_collector.save_dataset(final_df, 'full_dataset.csv')
    
    # Create training/test split
    from sklearn.model_selection import train_test_split
    
    # Separate features and labels
    feature_columns = [col for col in final_df.columns if col not in ['label', 'label_text', 'category', 'complexity', 'code']]
    X = final_df[feature_columns]
    y = final_df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    # Create training dataset
    train_df = X_train.copy()
    train_df['label'] = y_train
    train_df['label_text'] = final_df.loc[y_train.index, 'label_text']
    
    # Create test dataset
    test_df = X_test.copy()
    test_df['label'] = y_test
    test_df['label_text'] = final_df.loc[y_test.index, 'label_text']
    
    # Save split datasets
    train_path = config.DATA_DIR / 'training_data.csv'
    test_path = config.DATA_DIR / 'test_data.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"💾 Training dataset saved to {train_path}")
    print(f"💾 Test dataset saved to {test_path}")
    
    # Step 6: Generate dataset report
    print("\n📊 Step 6: Generating dataset report")
    print("-" * 40)
    
    generate_dataset_report(final_df, feature_columns, config.DATA_DIR)
    
    print("\n✅ Dataset creation completed successfully!")
    print("=" * 60)
    
    return final_df, feature_columns

def generate_dataset_report(df: pd.DataFrame, feature_columns: list, data_dir: Path):
    """Generate a comprehensive dataset report.
    
    Args:
        df: Complete dataset
        feature_columns: List of feature column names
        data_dir: Directory to save the report
    """
    report_path = data_dir / 'dataset_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Python Code Classifier - Dataset Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics
        f.write("📊 BASIC STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Human-written samples: {len(df[df['label'] == 0])}\n")
        f.write(f"AI-generated samples: {len(df[df['label'] == 1])}\n")
        f.write(f"Total features: {len(feature_columns)}\n\n")
        
        # Label distribution
        f.write("🏷️ LABEL DISTRIBUTION\n")
        f.write("-" * 30 + "\n")
        label_counts = df['label_text'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"{label}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Category distribution
        f.write("📂 CATEGORY DISTRIBUTION\n")
        f.write("-" * 30 + "\n")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"{category}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Complexity distribution
        f.write("🔧 COMPLEXITY DISTRIBUTION\n")
        f.write("-" * 30 + "\n")
        complexity_counts = df['complexity'].value_counts()
        for complexity, count in complexity_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"{complexity}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Feature statistics
        f.write("📈 FEATURE STATISTICS\n")
        f.write("-" * 30 + "\n")
        feature_stats = df[feature_columns].describe()
        f.write(feature_stats.to_string())
        f.write("\n\n")
        
        # Feature correlation with labels (only numeric features)
        f.write("🔗 TOP FEATURES BY CORRELATION WITH LABELS\n")
        f.write("-" * 30 + "\n")
        numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 0:
            correlations = df[numeric_features].corrwith(df['label']).abs().sort_values(ascending=False)
            for feature, corr in correlations.head(10).items():
                if not pd.isna(corr):
                    f.write(f"{feature}: {corr:.3f}\n")
        else:
            f.write("No numeric features found for correlation analysis.\n")
        f.write("\n")
        
        # Missing values
        f.write("❓ MISSING VALUES\n")
        f.write("-" * 30 + "\n")
        missing_values = df[feature_columns].isnull().sum()
        missing_features = missing_values[missing_values > 0]
        if len(missing_features) > 0:
            for feature, count in missing_features.items():
                percentage = (count / len(df)) * 100
                f.write(f"{feature}: {count} ({percentage:.1f}%)\n")
        else:
            f.write("No missing values found.\n")
        f.write("\n")
        
        # Feature list
        f.write("📋 COMPLETE FEATURE LIST\n")
        f.write("-" * 30 + "\n")
        for i, feature in enumerate(feature_columns, 1):
            f.write(f"{i:2d}. {feature}\n")
    
    print(f"📄 Dataset report saved to {report_path}")

def main():
    """Main function to run dataset creation."""
    parser = argparse.ArgumentParser(description='Create training dataset for Python Code Classifier')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Create the dataset
        dataset, features = create_training_dataset()
        
        if args.verbose:
            print("\n🔍 DATASET PREVIEW")
            print("-" * 40)
            print(dataset.head())
            print(f"\nDataset shape: {dataset.shape}")
            print(f"Features: {len(features)}")
            print(f"Samples: {len(dataset)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
