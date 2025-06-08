"""
Data Collection Module for Python Code Classifier

This module handles the collection and generation of training data for the code classifier.
It creates both AI-generated and human-written Python code samples.
"""

import os
import pandas as pd
import requests
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple
import ast
import re

class DataCollector:
    """Handles data collection for training the code classifier."""
    
    def __init__(self, data_dir: Path):
        """Initialize the data collector.
        
        Args:
            data_dir: Directory to store collected data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / 'raw').mkdir(exist_ok=True)
        (self.data_dir / 'processed').mkdir(exist_ok=True)
        (self.data_dir / 'human_samples').mkdir(exist_ok=True)
        (self.data_dir / 'ai_samples').mkdir(exist_ok=True)
    
    def create_human_written_samples(self) -> List[Dict]:
        """Create a collection of human-written Python code samples.
        
        Returns:
            List of dictionaries containing code samples and metadata
        """
        print("📝 Creating human-written code samples...")
        
        human_samples = [
            # Basic algorithms
            {
                'code': '''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Test the function
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = bubble_sort(numbers.copy())
print(f"Original: {numbers}")
print(f"Sorted: {sorted_numbers}")''',
                'category': 'algorithm',
                'complexity': 'medium',
                'author_type': 'human'
            },
            
            {
                'code': '''class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a, b):
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def get_history(self):
        return self.history

calc = Calculator()
print(calc.add(5, 3))
print(calc.subtract(10, 4))
print("History:", calc.get_history())''',
                'category': 'class',
                'complexity': 'medium',
                'author_type': 'human'
            },
            
            {
                'code': '''import random

def guess_number_game():
    number = random.randint(1, 100)
    attempts = 0
    max_attempts = 7
    
    print("Welcome to the Number Guessing Game!")
    print(f"I'm thinking of a number between 1 and 100.")
    print(f"You have {max_attempts} attempts to guess it.")
    
    while attempts < max_attempts:
        try:
            guess = int(input("Enter your guess: "))
            attempts += 1
            
            if guess == number:
                print(f"Congratulations! You guessed it in {attempts} attempts!")
                return True
            elif guess < number:
                print("Too low!")
            else:
                print("Too high!")
                
            print(f"Attempts remaining: {max_attempts - attempts}")
            
        except ValueError:
            print("Please enter a valid number!")
    
    print(f"Game over! The number was {number}")
    return False

if __name__ == "__main__":
    guess_number_game()''',
                'category': 'game',
                'complexity': 'high',
                'author_type': 'human'
            },
            
            {
                'code': '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_iterative(n):
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Compare both methods
n = 10
print(f"Fibonacci({n}) recursive: {fibonacci(n)}")
print(f"Fibonacci({n}) iterative: {fibonacci_iterative(n)}")''',
                'category': 'algorithm',
                'complexity': 'medium',
                'author_type': 'human'
            },
            
            {
                'code': '''def read_file_safely(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except PermissionError:
        print(f"Error: Permission denied to read '{filename}'.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Example usage
content = read_file_safely("example.txt")
if content:
    print("File content:", content[:100])  # First 100 characters
else:
    print("Failed to read file.")''',
                'category': 'file_handling',
                'complexity': 'medium',
                'author_type': 'human'
            }
        ]
        
        # Add more basic samples
        basic_samples = [
            {
                'code': '''# Simple hello world
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))''',
                'category': 'basic',
                'complexity': 'low',
                'author_type': 'human'
            },
            
            {
                'code': '''# List comprehension example
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_squares = [x**2 for x in numbers if x % 2 == 0]
print(even_squares)''',
                'category': 'basic',
                'complexity': 'low',
                'author_type': 'human'
            },
            
            {
                'code': '''# Dictionary operations
student_grades = {
    "Alice": 85,
    "Bob": 92,
    "Charlie": 78,
    "Diana": 96
}

# Find average grade
average = sum(student_grades.values()) / len(student_grades)
print(f"Average grade: {average:.2f}")

# Find top student
top_student = max(student_grades, key=student_grades.get)
print(f"Top student: {top_student} with {student_grades[top_student]}")''',
                'category': 'data_structures',
                'complexity': 'medium',
                'author_type': 'human'
            }
        ]
        
        human_samples.extend(basic_samples)
        
        # Save samples to files
        for i, sample in enumerate(human_samples):
            filename = self.data_dir / 'human_samples' / f'human_{i+1:03d}.py'
            with open(filename, 'w') as f:
                f.write(sample['code'])
        
        print(f"✅ Created {len(human_samples)} human-written samples")
        return human_samples
    
    def create_ai_generated_samples(self) -> List[Dict]:
        """Create AI-generated Python code samples.
        
        Returns:
            List of dictionaries containing AI-generated code samples
        """
        print("🤖 Creating AI-generated code samples...")
        
        # These samples simulate typical AI-generated code patterns
        ai_samples = [
            {
                'code': '''def calculate_factorial(n):
    """
    Calculate the factorial of a given number.
    
    Args:
        n (int): The number to calculate factorial for
        
    Returns:
        int: The factorial of n
    """
    if n == 0 or n == 1:
        return 1
    else:
        return n * calculate_factorial(n - 1)

def main():
    """Main function to demonstrate factorial calculation."""
    number = 5
    result = calculate_factorial(number)
    print(f"The factorial of {number} is {result}")

if __name__ == "__main__":
    main()''',
                'category': 'algorithm',
                'complexity': 'medium',
                'author_type': 'ai'
            },
            
            {
                'code': '''class BankAccount:
    """
    A class to represent a bank account.
    
    Attributes:
        account_number (str): The account number
        balance (float): The current balance
        account_holder (str): The name of the account holder
    """
    
    def __init__(self, account_number: str, account_holder: str, initial_balance: float = 0.0):
        """
        Initialize a new bank account.
        
        Args:
            account_number (str): The account number
            account_holder (str): The name of the account holder
            initial_balance (float): The initial balance (default: 0.0)
        """
        self.account_number = account_number
        self.account_holder = account_holder
        self.balance = initial_balance
    
    def deposit(self, amount: float) -> bool:
        """
        Deposit money into the account.
        
        Args:
            amount (float): The amount to deposit
            
        Returns:
            bool: True if successful, False otherwise
        """
        if amount > 0:
            self.balance += amount
            return True
        return False
    
    def withdraw(self, amount: float) -> bool:
        """
        Withdraw money from the account.
        
        Args:
            amount (float): The amount to withdraw
            
        Returns:
            bool: True if successful, False otherwise
        """
        if amount > 0 and amount <= self.balance:
            self.balance -= amount
            return True
        return False
    
    def get_balance(self) -> float:
        """
        Get the current balance.
        
        Returns:
            float: The current balance
        """
        return self.balance''',
                'category': 'class',
                'complexity': 'high',
                'author_type': 'ai'
            },
            
            {
                'code': '''def binary_search(arr, target):
    """
    Perform binary search on a sorted array.
    
    Args:
        arr (list): Sorted list to search in
        target: Element to search for
        
    Returns:
        int: Index of target if found, -1 otherwise
    """
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def test_binary_search():
    """Test function for binary search."""
    test_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    test_target = 7
    
    result = binary_search(test_array, test_target)
    
    if result != -1:
        print(f"Element {test_target} found at index {result}")
    else:
        print(f"Element {test_target} not found in the array")

if __name__ == "__main__":
    test_binary_search()''',
                'category': 'algorithm',
                'complexity': 'medium',
                'author_type': 'ai'
            }
        ]
        
        # Add more AI-style samples with typical AI patterns
        ai_style_samples = [
            {
                'code': '''def process_data(data_list):
    """
    Process a list of data items.
    
    Args:
        data_list (list): List of data items to process
        
    Returns:
        list: Processed data items
    """
    processed_items = []
    
    for item in data_list:
        if isinstance(item, (int, float)):
            processed_item = item * 2
        elif isinstance(item, str):
            processed_item = item.upper()
        else:
            processed_item = str(item)
        
        processed_items.append(processed_item)
    
    return processed_items

def main():
    """Main execution function."""
    sample_data = [1, 2.5, "hello", True, None, "world"]
    result = process_data(sample_data)
    print("Processed data:", result)

if __name__ == "__main__":
    main()''',
                'category': 'data_processing',
                'complexity': 'medium',
                'author_type': 'ai'
            },
            
            {
                'code': '''def validate_email(email: str) -> bool:
    """
    Validate an email address using basic pattern matching.
    
    Args:
        email (str): Email address to validate
        
    Returns:
        bool: True if email is valid, False otherwise
    """
    import re
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def test_email_validation():
    """Test the email validation function."""
    test_emails = [
        "user@example.com",
        "invalid.email",
        "test@domain.co.uk",
        "not_an_email",
        "another@test.org"
    ]
    
    for email in test_emails:
        is_valid = validate_email(email)
        status = "Valid" if is_valid else "Invalid"
        print(f"{email}: {status}")

if __name__ == "__main__":
    test_email_validation()''',
                'category': 'validation',
                'complexity': 'medium',
                'author_type': 'ai'
            }
        ]
        
        ai_samples.extend(ai_style_samples)
        
        # Save samples to files
        for i, sample in enumerate(ai_samples):
            filename = self.data_dir / 'ai_samples' / f'ai_{i+1:03d}.py'
            with open(filename, 'w') as f:
                f.write(sample['code'])
        
        print(f"✅ Created {len(ai_samples)} AI-generated samples")
        return ai_samples
    
    def combine_datasets(self, human_samples: List[Dict], ai_samples: List[Dict]) -> pd.DataFrame:
        """Combine human and AI samples into a single dataset.
        
        Args:
            human_samples: List of human-written code samples
            ai_samples: List of AI-generated code samples
            
        Returns:
            Combined dataset as pandas DataFrame
        """
        print("🔄 Combining datasets...")
        
        all_samples = []
        
        # Add human samples
        for sample in human_samples:
            all_samples.append({
                'code': sample['code'],
                'label': 0,  # 0 for human-written
                'label_text': 'human-written',
                'category': sample.get('category', 'unknown'),
                'complexity': sample.get('complexity', 'unknown'),
                'length': len(sample['code']),
                'lines': len(sample['code'].split('\n'))
            })
        
        # Add AI samples
        for sample in ai_samples:
            all_samples.append({
                'code': sample['code'],
                'label': 1,  # 1 for AI-generated
                'label_text': 'ai-generated',
                'category': sample.get('category', 'unknown'),
                'complexity': sample.get('complexity', 'unknown'),
                'length': len(sample['code']),
                'lines': len(sample['code'].split('\n'))
            })
        
        # Create DataFrame
        df = pd.DataFrame(all_samples)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Combined dataset created with {len(df)} samples")
        print(f"   - Human-written: {len(human_samples)} samples")
        print(f"   - AI-generated: {len(ai_samples)} samples")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = 'training_data.csv'):
        """Save the dataset to a CSV file.
        
        Args:
            df: Dataset to save
            filename: Name of the file to save
        """
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"💾 Dataset saved to {filepath}")
        
        # Print dataset statistics
        print("\n📊 Dataset Statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Human-written: {len(df[df['label'] == 0])}")
        print(f"AI-generated: {len(df[df['label'] == 1])}")
        if 'length' in df.columns:
            print(f"Average code length: {df['length'].mean():.1f} characters")
        if 'lines' in df.columns:
            print(f"Average lines of code: {df['lines'].mean():.1f}")
        if 'total_characters' in df.columns:
            print(f"Average code length: {df['total_characters'].mean():.1f} characters")
        if 'total_lines' in df.columns:
            print(f"Average lines of code: {df['total_lines'].mean():.1f}")
        
        return filepath
