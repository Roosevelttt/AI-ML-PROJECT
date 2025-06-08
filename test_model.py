"""
Model Testing Script for Python Code Classifier

This script tests the trained model with sample code to verify it's working correctly.
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.model_predictor import CodeClassifierPredictor
from config import Config

def test_sample_codes():
    """Test the model with sample code snippets."""
    
    # Sample human-written code
    human_code_samples = [
        '''def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))''',
        
        '''import json

def load_config(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return {}

config = load_config('config.json')
print(config.get('debug', False))''',
        
        '''class Stack:
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
print(s.pop())'''
    ]
    
    # Sample AI-generated code
    ai_code_samples = [
        '''def calculate_factorial(n: int) -> int:
    """
    Calculate the factorial of a given number.
    
    Args:
        n (int): The number to calculate factorial for
        
    Returns:
        int: The factorial of n
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0 or n == 1:
        return 1
    else:
        return n * calculate_factorial(n - 1)

def main() -> None:
    """Main function to demonstrate factorial calculation."""
    try:
        number = 5
        result = calculate_factorial(number)
        print(f"The factorial of {number} is {result}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()''',
        
        '''class DataProcessor:
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
    main()'''
    ]
    
    return human_code_samples, ai_code_samples

def run_model_tests():
    """Run comprehensive model tests."""
    print("🧪 Starting Model Testing")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config = Config()
        
        # Check if model exists
        model_path = config.MODEL_DIR / 'best_model.joblib'
        if not model_path.exists():
            print("❌ Trained model not found!")
            print("Please run 'python train_model.py' first to train the model.")
            return False
        
        # Initialize predictor
        print("🔧 Loading trained model...")
        predictor = CodeClassifierPredictor(config.MODEL_DIR)
        
        # Validate setup
        validation = predictor.validate_prediction_setup()
        if not validation['ready_for_prediction']:
            print("❌ Model setup validation failed:")
            for component, status in validation.items():
                print(f"  {component}: {'✅' if status else '❌'}")
            return False
        
        print("✅ Model loaded successfully!")
        
        # Get model info
        model_info = predictor.get_model_info()
        print(f"📊 Model: {model_info.get('model_name', 'Unknown')}")
        print(f"📊 Features: {model_info.get('feature_count', 0)}")
        
        # Get test samples
        human_samples, ai_samples = test_sample_codes()
        
        print("\n🔍 Testing Human-Written Code Samples")
        print("-" * 40)
        
        human_correct = 0
        for i, code in enumerate(human_samples, 1):
            print(f"\n📝 Human Sample {i}:")
            result = predictor.predict(code)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            prediction = result['prediction_text']
            confidence = result['confidence']
            
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.3f}")
            
            if prediction == 'human-written':
                print("   ✅ Correct!")
                human_correct += 1
            else:
                print("   ❌ Incorrect!")
        
        print("\n🤖 Testing AI-Generated Code Samples")
        print("-" * 40)
        
        ai_correct = 0
        for i, code in enumerate(ai_samples, 1):
            print(f"\n🤖 AI Sample {i}:")
            result = predictor.predict(code)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            prediction = result['prediction_text']
            confidence = result['confidence']
            
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.3f}")
            
            if prediction == 'ai-generated':
                print("   ✅ Correct!")
                ai_correct += 1
            else:
                print("   ❌ Incorrect!")
        
        # Calculate accuracy
        total_samples = len(human_samples) + len(ai_samples)
        total_correct = human_correct + ai_correct
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        print("\n📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"Human samples correct: {human_correct}/{len(human_samples)}")
        print(f"AI samples correct: {ai_correct}/{len(ai_samples)}")
        print(f"Overall accuracy: {accuracy:.3f} ({total_correct}/{total_samples})")
        
        if accuracy >= 0.7:
            print("🎉 Model performance is good!")
        elif accuracy >= 0.5:
            print("⚠️  Model performance is moderate.")
        else:
            print("❌ Model performance needs improvement.")
        
        # Test detailed analysis
        print("\n🔍 DETAILED FEATURE ANALYSIS")
        print("-" * 40)
        
        sample_code = human_samples[0]
        analysis = predictor.analyze_code_features(sample_code)
        
        if 'error' not in analysis:
            print("Sample code analysis:")
            print(f"  Prediction: {analysis['prediction']['prediction_text']}")
            print(f"  Confidence: {analysis['prediction']['confidence']:.3f}")
            
            features = analysis['feature_analysis']
            print(f"  Basic features:")
            print(f"    Lines: {features['basic']['total_lines']}")
            print(f"    Characters: {features['basic']['total_characters']}")
            print(f"  Syntactic features:")
            print(f"    Functions: {features['syntactic']['num_functions']}")
            print(f"    Classes: {features['syntactic']['num_classes']}")
            print(f"  Documentation features:")
            print(f"    Docstrings: {features['documentation']['num_docstrings']}")
            print(f"    Type hints: {features['documentation']['type_hints_count']}")
        
        print("\n✅ Model testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    success = run_model_tests()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
