"""
Model Evaluation Script for AI/ML Code Classifier

This script evaluates the trained model's performance on various code samples
and generates a comprehensive evaluation report.
"""

import requests
import json
import sys
from pathlib import Path
import time

def test_classification_api():
    """Test the classification API with various code samples."""
    
    print("🧪 Testing AI/ML Code Classifier API")
    print("=" * 50)
    
    # Test samples with known labels
    test_samples = [
        # Human-written samples
        {
            'code': '''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

numbers = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(numbers))''',
            'expected': 'human-written',
            'category': 'algorithm'
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
            'expected': 'human-written',
            'category': 'file_handling'
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

s = Stack()
s.push(1)
print(s.pop())''',
            'expected': 'human-written',
            'category': 'data_structures'
        },
        
        # AI-generated samples
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
            'expected': 'ai-generated',
            'category': 'algorithm'
        },
        
        {
            'code': '''class DataProcessor:
    """
    A class for processing various types of data.
    
    This class provides methods to clean, validate, and transform data
    according to specified rules and formats.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the data processor.
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = config or {}
        self.processed_count = 0
    
    def process_data(self, data: list) -> list:
        """
        Process a list of data items.
        
        Args:
            data (list): List of data items to process
            
        Returns:
            list: Processed data items
            
        Raises:
            TypeError: If data is not a list
        """
        if not isinstance(data, list):
            raise TypeError("Data must be a list")
        
        processed_items = []
        
        for item in data:
            if isinstance(item, (int, float)):
                processed_item = item * 2
            elif isinstance(item, str):
                processed_item = item.upper()
            else:
                processed_item = str(item)
            
            processed_items.append(processed_item)
            self.processed_count += 1
        
        return processed_items

def main() -> None:
    """Main execution function."""
    processor = DataProcessor()
    sample_data = [1, 2.5, "hello", True, None]
    result = processor.process_data(sample_data)
    print(f"Processed data: {result}")
    print(f"Items processed: {processor.processed_count}")

if __name__ == "__main__":
    main()''',
            'expected': 'ai-generated',
            'category': 'data_processing'
        }
    ]
    
    # Test API availability
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code != 200:
            print("❌ API is not available")
            return False
        
        health_data = response.json()
        print(f"✅ API is healthy")
        print(f"📊 Model: {health_data['model_info']['model_name']}")
        print(f"📊 Features: {health_data['model_info']['feature_count']}")
        
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Please make sure the Flask application is running on http://localhost:5000")
        return False
    
    # Run tests
    print(f"\n🔍 Testing {len(test_samples)} code samples...")
    print("-" * 50)
    
    results = []
    correct_predictions = 0
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\n📝 Test {i}: {sample['category']} ({sample['expected']})")
        
        try:
            # Make prediction
            response = requests.post(
                'http://localhost:5000/classify',
                data={'code': sample['code']},
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"❌ API error: {response.status_code}")
                continue
            
            result = response.json()
            
            if not result.get('success', False):
                print(f"❌ Classification failed: {result.get('error', 'Unknown error')}")
                continue
            
            prediction = result['prediction']
            confidence = result['confidence']
            model_name = result.get('model_name', 'Unknown')
            
            # Check if prediction is correct
            is_correct = prediction == sample['expected']
            if is_correct:
                correct_predictions += 1
            
            # Store result
            results.append({
                'test_id': i,
                'category': sample['category'],
                'expected': sample['expected'],
                'predicted': prediction,
                'confidence': confidence,
                'correct': is_correct,
                'features': result.get('features', {})
            })
            
            # Print result
            status = "✅ Correct" if is_correct else "❌ Incorrect"
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   {status}")
            
            # Print key features
            features = result.get('features', {})
            print(f"   Features: {features.get('lines_of_code', 0)} lines, "
                  f"{features.get('num_docstrings', 0)} docstrings, "
                  f"{features.get('type_hints_count', 0)} type hints")
            
        except Exception as e:
            print(f"❌ Error testing sample {i}: {e}")
    
    # Calculate overall accuracy
    total_tests = len(results)
    accuracy = correct_predictions / total_tests if total_tests > 0 else 0
    
    print(f"\n📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Breakdown by category
    categories = {}
    for result in results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = {'total': 0, 'correct': 0}
        categories[cat]['total'] += 1
        if result['correct']:
            categories[cat]['correct'] += 1
    
    print(f"\n📈 Accuracy by Category:")
    for category, stats in categories.items():
        cat_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"   {category}: {cat_accuracy:.3f} ({stats['correct']}/{stats['total']})")
    
    # Breakdown by expected label
    labels = {}
    for result in results:
        label = result['expected']
        if label not in labels:
            labels[label] = {'total': 0, 'correct': 0}
        labels[label]['total'] += 1
        if result['correct']:
            labels[label]['correct'] += 1
    
    print(f"\n🏷️ Accuracy by Label:")
    for label, stats in labels.items():
        label_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"   {label}: {label_accuracy:.3f} ({stats['correct']}/{stats['total']})")
    
    # Confidence analysis
    confidences = [r['confidence'] for r in results]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"\n🎯 Confidence Analysis:")
    print(f"   Average confidence: {avg_confidence:.3f}")
    print(f"   Min confidence: {min(confidences):.3f}")
    print(f"   Max confidence: {max(confidences):.3f}")
    
    # Performance assessment
    print(f"\n🎉 Performance Assessment:")
    if accuracy >= 0.9:
        print("   🌟 Excellent performance!")
    elif accuracy >= 0.8:
        print("   ✅ Good performance!")
    elif accuracy >= 0.7:
        print("   👍 Acceptable performance")
    elif accuracy >= 0.6:
        print("   ⚠️  Moderate performance")
    else:
        print("   ❌ Poor performance - needs improvement")
    
    return accuracy >= 0.7

def main():
    """Main function."""
    print("🚀 Starting Model Evaluation")
    print("=" * 60)
    
    success = test_classification_api()
    
    if success:
        print("\n✅ Model evaluation completed successfully!")
    else:
        print("\n❌ Model evaluation failed or performance is poor")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
