"""
Dataset Expansion Script for AI/ML Code Classifier

This script adds more code samples to increase the dataset size for better training.
It includes more diverse examples of both human-written and AI-generated code.
"""

import pandas as pd
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.data_collector import DataCollector
from utils.feature_extractor import CodeFeatureExtractor
from config import Config

def create_additional_human_samples():
    """Create additional human-written code samples."""
    return [
        {
            'code': '''# Quick sort implementation
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)

# Test
test_array = [3, 6, 8, 10, 1, 2, 1]
print(quicksort(test_array))''',
            'category': 'algorithm',
            'complexity': 'medium',
            'author_type': 'human'
        },
        
        {
            'code': '''import json

def load_config(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return {}

config = load_config('config.json')
print(config.get('debug', False))''',
            'category': 'file_handling',
            'complexity': 'low',
            'author_type': 'human'
        },
        
        {
            'code': '''class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

s = Stack()
s.push(1)
s.push(2)
print(s.pop())''',
            'category': 'data_structures',
            'complexity': 'medium',
            'author_type': 'human'
        },
        
        {
            'code': '''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [i for i in range(2, 50) if is_prime(i)]
print(primes)''',
            'category': 'algorithm',
            'complexity': 'medium',
            'author_type': 'human'
        },
        
        {
            'code': '''# Simple web scraper
import requests
from bs4 import BeautifulSoup

def get_title(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.title.string
    except:
        return None

title = get_title('https://example.com')
if title:
    print(f"Title: {title}")''',
            'category': 'web_scraping',
            'complexity': 'high',
            'author_type': 'human'
        }
    ]

def create_additional_ai_samples():
    """Create additional AI-generated code samples."""
    return [
        {
            'code': '''def merge_sort(arr: list) -> list:
    """
    Implement merge sort algorithm.
    
    Args:
        arr (list): List of elements to sort
        
    Returns:
        list: Sorted list
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left: list, right: list) -> list:
    """
    Merge two sorted lists.
    
    Args:
        left (list): First sorted list
        right (list): Second sorted list
        
    Returns:
        list: Merged sorted list
    """
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

def main():
    """Main function to demonstrate merge sort."""
    test_array = [64, 34, 25, 12, 22, 11, 90]
    sorted_array = merge_sort(test_array)
    print(f"Original: {test_array}")
    print(f"Sorted: {sorted_array}")

if __name__ == "__main__":
    main()''',
            'category': 'algorithm',
            'complexity': 'high',
            'author_type': 'ai'
        },
        
        {
            'code': '''class ConfigManager:
    """
    A class to manage application configuration.
    
    This class provides methods to load, save, and access configuration
    settings from various sources including files and environment variables.
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_file (str): Path to the configuration file
        """
        self.config_file = config_file
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """
        Load configuration from file.
        
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            json.JSONDecodeError: If configuration file is invalid JSON
        """
        import json
        import os
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as file:
                    self.config = json.load(file)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"Invalid JSON in config file: {e}")
        else:
            self.config = {}
    
    def get(self, key: str, default=None):
        """
        Get configuration value by key.
        
        Args:
            key (str): Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value) -> None:
        """
        Set configuration value.
        
        Args:
            key (str): Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def save_config(self) -> None:
        """
        Save configuration to file.
        
        Raises:
            IOError: If unable to write to configuration file
        """
        import json
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as file:
                json.dump(self.config, file, indent=4, ensure_ascii=False)
        except IOError as e:
            raise IOError(f"Unable to save config file: {e}")''',
            'category': 'class',
            'complexity': 'high',
            'author_type': 'ai'
        },
        
        {
            'code': '''def calculate_statistics(data: list) -> dict:
    """
    Calculate basic statistics for a list of numbers.
    
    Args:
        data (list): List of numeric values
        
    Returns:
        dict: Dictionary containing statistical measures
        
    Raises:
        ValueError: If data is empty or contains non-numeric values
    """
    if not data:
        raise ValueError("Data list cannot be empty")
    
    # Validate that all elements are numeric
    for item in data:
        if not isinstance(item, (int, float)):
            raise ValueError(f"All elements must be numeric, found: {type(item)}")
    
    # Calculate statistics
    n = len(data)
    mean = sum(data) / n
    
    # Calculate variance and standard deviation
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = variance ** 0.5
    
    # Find min and max
    minimum = min(data)
    maximum = max(data)
    
    # Calculate median
    sorted_data = sorted(data)
    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        median = sorted_data[n // 2]
    
    return {
        'count': n,
        'mean': mean,
        'median': median,
        'variance': variance,
        'std_dev': std_dev,
        'min': minimum,
        'max': maximum,
        'range': maximum - minimum
    }

def main():
    """Main function to demonstrate statistics calculation."""
    sample_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    try:
        stats = calculate_statistics(sample_data)
        print("Statistical Analysis:")
        for key, value in stats.items():
            print(f"{key.capitalize()}: {value:.2f}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()''',
            'category': 'data_processing',
            'complexity': 'medium',
            'author_type': 'ai'
        }
    ]

def expand_dataset():
    """Expand the existing dataset with additional samples."""
    print("🚀 Expanding dataset with additional samples...")
    print("=" * 60)
    
    # Initialize components
    config = Config()
    data_collector = DataCollector(config.DATA_DIR)
    feature_extractor = CodeFeatureExtractor()
    
    # Load existing dataset
    existing_path = config.DATA_DIR / 'full_dataset.csv'
    if not existing_path.exists():
        print("❌ No existing dataset found. Please run create_dataset.py first.")
        return False
    
    existing_df = pd.read_csv(existing_path)
    print(f"📊 Loaded existing dataset with {len(existing_df)} samples")
    
    # Create additional samples
    print("\n📝 Creating additional samples...")
    additional_human = create_additional_human_samples()
    additional_ai = create_additional_ai_samples()
    
    print(f"✅ Created {len(additional_human)} additional human samples")
    print(f"✅ Created {len(additional_ai)} additional AI samples")
    
    # Combine new samples
    new_samples = data_collector.combine_datasets(additional_human, additional_ai)
    
    # Extract features for new samples
    print("\n🔍 Extracting features for new samples...")
    new_features_df = feature_extractor.extract_features_batch(new_samples['code'].tolist())
    
    # Add labels and metadata
    final_new_df = new_features_df.copy()
    final_new_df['label'] = new_samples['label']
    final_new_df['label_text'] = new_samples['label_text']
    final_new_df['category'] = new_samples['category']
    final_new_df['complexity'] = new_samples['complexity']
    final_new_df['code'] = new_samples['code']
    
    # Combine with existing dataset
    print("\n🔗 Combining with existing dataset...")
    
    # Ensure columns match
    existing_columns = set(existing_df.columns)
    new_columns = set(final_new_df.columns)
    
    if existing_columns != new_columns:
        print("⚠️  Column mismatch detected. Aligning columns...")
        # Add missing columns with default values
        for col in existing_columns - new_columns:
            final_new_df[col] = 0 if existing_df[col].dtype in ['int64', 'float64'] else ''
        for col in new_columns - existing_columns:
            existing_df[col] = 0 if final_new_df[col].dtype in ['int64', 'float64'] else ''
        
        # Reorder columns to match
        final_new_df = final_new_df[existing_df.columns]
    
    # Combine datasets
    expanded_df = pd.concat([existing_df, final_new_df], ignore_index=True)
    
    # Shuffle the combined dataset
    expanded_df = expanded_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Expanded dataset now has {len(expanded_df)} samples")
    print(f"   - Human-written: {len(expanded_df[expanded_df['label'] == 0])}")
    print(f"   - AI-generated: {len(expanded_df[expanded_df['label'] == 1])}")
    
    # Save expanded dataset
    print("\n💾 Saving expanded dataset...")
    expanded_path = config.DATA_DIR / 'expanded_dataset.csv'
    expanded_df.to_csv(expanded_path, index=False)
    print(f"💾 Expanded dataset saved to {expanded_path}")
    
    # Update the main dataset files
    data_collector.save_dataset(expanded_df, 'full_dataset.csv')
    
    # Create new train/test split
    from sklearn.model_selection import train_test_split
    
    feature_columns = [col for col in expanded_df.columns if col not in ['label', 'label_text', 'category', 'complexity', 'code']]
    X = expanded_df[feature_columns]
    y = expanded_df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    # Create new training dataset
    train_df = X_train.copy()
    train_df['label'] = y_train
    train_df['label_text'] = expanded_df.loc[y_train.index, 'label_text']
    
    # Create new test dataset
    test_df = X_test.copy()
    test_df['label'] = y_test
    test_df['label_text'] = expanded_df.loc[y_test.index, 'label_text']
    
    # Save new split datasets
    train_path = config.DATA_DIR / 'training_data.csv'
    test_path = config.DATA_DIR / 'test_data.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"💾 Updated training dataset: {len(train_df)} samples")
    print(f"💾 Updated test dataset: {len(test_df)} samples")
    
    print("\n✅ Dataset expansion completed successfully!")
    print("=" * 60)
    
    return True

def main():
    """Main function to run dataset expansion."""
    try:
        success = expand_dataset()
        return success
    except Exception as e:
        print(f"❌ Error expanding dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
