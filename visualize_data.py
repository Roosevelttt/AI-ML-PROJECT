"""
Data Visualization Script for Python Code Classifier

This script creates visualizations to help understand the dataset characteristics
and feature distributions between AI-generated and human-written code.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from config import Config

def load_dataset():
    """Load the training dataset."""
    config = Config()
    dataset_path = config.DATA_DIR / 'full_dataset.csv'
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please run 'python create_dataset.py' first.")
        return None
    
    df = pd.read_csv(dataset_path)
    print(f"✅ Loaded dataset with {len(df)} samples and {len(df.columns)} columns")
    return df

def create_visualizations(df):
    """Create various visualizations of the dataset."""
    print("📊 Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Label distribution
    plt.subplot(3, 4, 1)
    label_counts = df['label_text'].value_counts()
    colors = ['#2E86AB', '#A23B72']
    plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('Label Distribution', fontsize=14, fontweight='bold')
    
    # 2. Category distribution
    plt.subplot(3, 4, 2)
    category_counts = df['category'].value_counts()
    plt.bar(range(len(category_counts)), category_counts.values, color='skyblue')
    plt.xticks(range(len(category_counts)), category_counts.index, rotation=45, ha='right')
    plt.title('Category Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    
    # 3. Complexity distribution
    plt.subplot(3, 4, 3)
    complexity_counts = df['complexity'].value_counts()
    plt.bar(complexity_counts.index, complexity_counts.values, color='lightcoral')
    plt.title('Complexity Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    
    # 4. Code length distribution by label
    plt.subplot(3, 4, 4)
    human_lengths = df[df['label'] == 0]['total_characters']
    ai_lengths = df[df['label'] == 1]['total_characters']
    plt.hist([human_lengths, ai_lengths], bins=8, alpha=0.7, 
             label=['Human-written', 'AI-generated'], color=['#2E86AB', '#A23B72'])
    plt.title('Code Length Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Total Characters')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 5. Lines of code distribution
    plt.subplot(3, 4, 5)
    human_lines = df[df['label'] == 0]['total_lines']
    ai_lines = df[df['label'] == 1]['total_lines']
    plt.hist([human_lines, ai_lines], bins=8, alpha=0.7,
             label=['Human-written', 'AI-generated'], color=['#2E86AB', '#A23B72'])
    plt.title('Lines of Code Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Total Lines')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 6. Documentation features comparison
    plt.subplot(3, 4, 6)
    doc_features = ['num_docstrings', 'type_hints_count', 'comment_lines']
    human_doc = df[df['label'] == 0][doc_features].mean()
    ai_doc = df[df['label'] == 1][doc_features].mean()
    
    x = np.arange(len(doc_features))
    width = 0.35
    plt.bar(x - width/2, human_doc, width, label='Human-written', color='#2E86AB')
    plt.bar(x + width/2, ai_doc, width, label='AI-generated', color='#A23B72')
    plt.xticks(x, doc_features, rotation=45, ha='right')
    plt.title('Documentation Features', fontsize=14, fontweight='bold')
    plt.ylabel('Average Count')
    plt.legend()
    
    # 7. Complexity features comparison
    plt.subplot(3, 4, 7)
    complexity_features = ['num_functions', 'num_classes', 'cyclomatic_complexity']
    human_complex = df[df['label'] == 0][complexity_features].mean()
    ai_complex = df[df['label'] == 1][complexity_features].mean()
    
    x = np.arange(len(complexity_features))
    plt.bar(x - width/2, human_complex, width, label='Human-written', color='#2E86AB')
    plt.bar(x + width/2, ai_complex, width, label='AI-generated', color='#A23B72')
    plt.xticks(x, complexity_features, rotation=45, ha='right')
    plt.title('Complexity Features', fontsize=14, fontweight='bold')
    plt.ylabel('Average Count')
    plt.legend()
    
    # 8. Style features comparison
    plt.subplot(3, 4, 8)
    style_features = ['snake_case_count', 'camel_case_count', 'space_indented_lines']
    human_style = df[df['label'] == 0][style_features].mean()
    ai_style = df[df['label'] == 1][style_features].mean()
    
    x = np.arange(len(style_features))
    plt.bar(x - width/2, human_style, width, label='Human-written', color='#2E86AB')
    plt.bar(x + width/2, ai_style, width, label='AI-generated', color='#A23B72')
    plt.xticks(x, style_features, rotation=45, ha='right')
    plt.title('Style Features', fontsize=14, fontweight='bold')
    plt.ylabel('Average Count')
    plt.legend()
    
    # 9. Top correlations with labels
    plt.subplot(3, 4, 9)
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col not in ['label']]
    correlations = df[numeric_features].corrwith(df['label']).abs().sort_values(ascending=False)
    top_corr = correlations.head(8)
    
    plt.barh(range(len(top_corr)), top_corr.values, color='orange')
    plt.yticks(range(len(top_corr)), top_corr.index)
    plt.title('Top Feature Correlations', fontsize=14, fontweight='bold')
    plt.xlabel('Absolute Correlation with Label')
    
    # 10. Feature distribution heatmap (sample)
    plt.subplot(3, 4, 10)
    sample_features = ['total_characters', 'total_lines', 'num_functions', 'num_docstrings', 
                      'type_hints_count', 'cyclomatic_complexity']
    corr_matrix = df[sample_features + ['label']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 11. Token diversity comparison
    plt.subplot(3, 4, 11)
    human_diversity = df[df['label'] == 0]['token_diversity']
    ai_diversity = df[df['label'] == 1]['token_diversity']
    plt.boxplot([human_diversity, ai_diversity], tick_labels=['Human-written', 'AI-generated'])
    plt.title('Token Diversity Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Token Diversity')
    
    # 12. Readability scores comparison
    plt.subplot(3, 4, 12)
    readability_features = ['flesch_reading_ease', 'flesch_kincaid_grade']
    human_read = df[df['label'] == 0][readability_features].mean()
    ai_read = df[df['label'] == 1][readability_features].mean()
    
    x = np.arange(len(readability_features))
    plt.bar(x - width/2, human_read, width, label='Human-written', color='#2E86AB')
    plt.bar(x + width/2, ai_read, width, label='AI-generated', color='#A23B72')
    plt.xticks(x, ['Flesch Reading\nEase', 'Flesch-Kincaid\nGrade'])
    plt.title('Readability Scores', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    config = Config()
    plot_path = config.DATA_DIR / 'dataset_visualization.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to {plot_path}")
    
    # Close the plot to free memory
    plt.close()

def print_dataset_insights(df):
    """Print key insights about the dataset."""
    print("\n🔍 DATASET INSIGHTS")
    print("=" * 50)
    
    # Basic statistics
    human_df = df[df['label'] == 0]
    ai_df = df[df['label'] == 1]
    
    print(f"📊 Sample Distribution:")
    print(f"   Human-written: {len(human_df)} samples ({len(human_df)/len(df)*100:.1f}%)")
    print(f"   AI-generated: {len(ai_df)} samples ({len(ai_df)/len(df)*100:.1f}%)")
    
    print(f"\n📏 Code Length Comparison:")
    print(f"   Human avg length: {human_df['total_characters'].mean():.1f} characters")
    print(f"   AI avg length: {ai_df['total_characters'].mean():.1f} characters")
    print(f"   Human avg lines: {human_df['total_lines'].mean():.1f} lines")
    print(f"   AI avg lines: {ai_df['total_lines'].mean():.1f} lines")
    
    print(f"\n📚 Documentation Comparison:")
    print(f"   Human avg docstrings: {human_df['num_docstrings'].mean():.1f}")
    print(f"   AI avg docstrings: {ai_df['num_docstrings'].mean():.1f}")
    print(f"   Human avg type hints: {human_df['type_hints_count'].mean():.1f}")
    print(f"   AI avg type hints: {ai_df['type_hints_count'].mean():.1f}")
    
    print(f"\n🔧 Complexity Comparison:")
    print(f"   Human avg functions: {human_df['num_functions'].mean():.1f}")
    print(f"   AI avg functions: {ai_df['num_functions'].mean():.1f}")
    print(f"   Human avg complexity: {human_df['cyclomatic_complexity'].mean():.1f}")
    print(f"   AI avg complexity: {ai_df['cyclomatic_complexity'].mean():.1f}")
    
    # Top distinguishing features
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col not in ['label']]
    correlations = df[numeric_features].corrwith(df['label']).abs().sort_values(ascending=False)
    
    print(f"\n🎯 Top Distinguishing Features:")
    for i, (feature, corr) in enumerate(correlations.head(5).items(), 1):
        print(f"   {i}. {feature}: {corr:.3f} correlation")

def main():
    """Main function to run data visualization."""
    print("🚀 Starting data visualization...")
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return False
    
    # Print insights
    print_dataset_insights(df)
    
    # Create visualizations
    try:
        create_visualizations(df)
        print("\n✅ Data visualization completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
