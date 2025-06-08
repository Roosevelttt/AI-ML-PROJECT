"""
Feature Extraction Module for Python Code Classifier

This module extracts various features from Python code that can help distinguish
between AI-generated and human-written code.
"""

import ast
import re
import textstat
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from collections import Counter

class CodeFeatureExtractor:
    """Extracts features from Python code for classification."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = []
    
    def extract_features(self, code: str) -> Dict[str, Any]:
        """Extract all features from a code sample.
        
        Args:
            code: Python code as string
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic text features
        features.update(self._extract_basic_features(code))
        
        # Syntactic features
        features.update(self._extract_syntactic_features(code))
        
        # Style features
        features.update(self._extract_style_features(code))
        
        # Complexity features
        features.update(self._extract_complexity_features(code))
        
        # Documentation features
        features.update(self._extract_documentation_features(code))
        
        return features
    
    def _extract_basic_features(self, code: str) -> Dict[str, Any]:
        """Extract basic text-based features.
        
        Args:
            code: Python code as string
            
        Returns:
            Dictionary of basic features
        """
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        return {
            'total_characters': len(code),
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'avg_line_length': np.mean([len(line) for line in lines]) if lines else 0,
            'max_line_length': max([len(line) for line in lines]) if lines else 0,
            'empty_line_ratio': (len(lines) - len(non_empty_lines)) / len(lines) if lines else 0,
            'whitespace_ratio': sum(1 for c in code if c.isspace()) / len(code) if code else 0
        }
    
    def _extract_syntactic_features(self, code: str) -> Dict[str, Any]:
        """Extract syntactic features using AST parsing.
        
        Args:
            code: Python code as string
            
        Returns:
            Dictionary of syntactic features
        """
        features = {
            'syntax_error': False,
            'num_functions': 0,
            'num_classes': 0,
            'num_imports': 0,
            'num_variables': 0,
            'num_if_statements': 0,
            'num_for_loops': 0,
            'num_while_loops': 0,
            'num_try_except': 0,
            'max_nesting_depth': 0,
            'num_return_statements': 0,
            'num_print_statements': 0
        }
        
        try:
            tree = ast.parse(code)
            
            # Count different node types
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    features['num_functions'] += 1
                elif isinstance(node, ast.ClassDef):
                    features['num_classes'] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    features['num_imports'] += 1
                elif isinstance(node, ast.Assign):
                    features['num_variables'] += 1
                elif isinstance(node, ast.If):
                    features['num_if_statements'] += 1
                elif isinstance(node, ast.For):
                    features['num_for_loops'] += 1
                elif isinstance(node, ast.While):
                    features['num_while_loops'] += 1
                elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                    features['num_try_except'] += 1
                elif isinstance(node, ast.Return):
                    features['num_return_statements'] += 1
                elif isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id == 'print':
                    features['num_print_statements'] += 1
            
            # Calculate nesting depth
            features['max_nesting_depth'] = self._calculate_nesting_depth(tree)
            
        except SyntaxError:
            features['syntax_error'] = True
        
        return features
    
    def _extract_style_features(self, code: str) -> Dict[str, Any]:
        """Extract code style features.
        
        Args:
            code: Python code as string
            
        Returns:
            Dictionary of style features
        """
        # Count different types of quotes
        single_quotes = code.count("'")
        double_quotes = code.count('"')
        
        # Count indentation patterns
        lines = code.split('\n')
        tab_indented = sum(1 for line in lines if line.startswith('\t'))
        space_indented = sum(1 for line in lines if re.match(r'^ +', line))
        
        # Count naming conventions
        snake_case_vars = len(re.findall(r'\b[a-z]+(_[a-z]+)*\b', code))
        camel_case_vars = len(re.findall(r'\b[a-z]+([A-Z][a-z]*)+\b', code))
        
        # Count comments
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        
        return {
            'single_quote_count': single_quotes,
            'double_quote_count': double_quotes,
            'quote_preference': 'single' if single_quotes > double_quotes else 'double',
            'tab_indented_lines': tab_indented,
            'space_indented_lines': space_indented,
            'indentation_preference': 'tabs' if tab_indented > space_indented else 'spaces',
            'snake_case_count': snake_case_vars,
            'camel_case_count': camel_case_vars,
            'naming_preference': 'snake_case' if snake_case_vars > camel_case_vars else 'camel_case',
            'comment_lines': comment_lines,
            'comment_ratio': comment_lines / len(lines) if lines else 0
        }
    
    def _extract_complexity_features(self, code: str) -> Dict[str, Any]:
        """Extract code complexity features.
        
        Args:
            code: Python code as string
            
        Returns:
            Dictionary of complexity features
        """
        # Use textstat for readability metrics
        features = {
            'flesch_reading_ease': textstat.flesch_reading_ease(code),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(code),
            'automated_readability_index': textstat.automated_readability_index(code),
            'coleman_liau_index': textstat.coleman_liau_index(code),
            'gunning_fog': textstat.gunning_fog(code)
        }
        
        # Calculate cyclomatic complexity (simplified)
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with']
        features['cyclomatic_complexity'] = sum(len(re.findall(rf'\b{keyword}\b', code)) for keyword in complexity_keywords)
        
        # Count unique tokens
        tokens = re.findall(r'\b\w+\b', code)
        features['unique_tokens'] = len(set(tokens))
        features['total_tokens'] = len(tokens)
        features['token_diversity'] = len(set(tokens)) / len(tokens) if tokens else 0
        
        return features
    
    def _extract_documentation_features(self, code: str) -> Dict[str, Any]:
        """Extract documentation-related features.
        
        Args:
            code: Python code as string
            
        Returns:
            Dictionary of documentation features
        """
        # Count docstrings
        docstring_pattern = r'""".*?"""'
        docstrings = re.findall(docstring_pattern, code, re.DOTALL)
        
        # Count type hints
        type_hints = len(re.findall(r':\s*[A-Za-z_][A-Za-z0-9_]*', code))
        
        # Count function annotations
        return_annotations = len(re.findall(r'->\s*[A-Za-z_][A-Za-z0-9_]*', code))
        
        # Check for main guard
        has_main_guard = 'if __name__ == "__main__"' in code
        
        return {
            'num_docstrings': len(docstrings),
            'total_docstring_length': sum(len(doc) for doc in docstrings),
            'avg_docstring_length': np.mean([len(doc) for doc in docstrings]) if docstrings else 0,
            'type_hints_count': type_hints,
            'return_annotations_count': return_annotations,
            'has_main_guard': has_main_guard,
            'documentation_ratio': (len(docstrings) + type_hints + return_annotations) / max(1, len(code.split('\n')))
        }
    
    def _calculate_nesting_depth(self, node, depth=0):
        """Calculate maximum nesting depth of AST.
        
        Args:
            node: AST node
            depth: Current depth
            
        Returns:
            Maximum nesting depth
        """
        max_depth = depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)):
                child_depth = self._calculate_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_nesting_depth(child, depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def extract_features_batch(self, code_samples: List[str]) -> pd.DataFrame:
        """Extract features from multiple code samples.
        
        Args:
            code_samples: List of code strings
            
        Returns:
            DataFrame with extracted features
        """
        print(f"🔍 Extracting features from {len(code_samples)} code samples...")
        
        features_list = []
        
        for i, code in enumerate(code_samples):
            if i % 10 == 0:
                print(f"   Processing sample {i+1}/{len(code_samples)}")
            
            try:
                features = self.extract_features(code)
                features_list.append(features)
            except Exception as e:
                print(f"   Warning: Error processing sample {i+1}: {e}")
                # Add default features for failed samples
                features_list.append(self._get_default_features())
        
        df = pd.DataFrame(features_list)
        
        # Store feature names for later use
        self.feature_names = df.columns.tolist()
        
        print(f"✅ Feature extraction completed. Extracted {len(df.columns)} features.")
        return df
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Get default feature values for failed extractions.
        
        Returns:
            Dictionary with default feature values
        """
        return {
            'total_characters': 0,
            'total_lines': 0,
            'non_empty_lines': 0,
            'avg_line_length': 0,
            'max_line_length': 0,
            'empty_line_ratio': 0,
            'whitespace_ratio': 0,
            'syntax_error': True,
            'num_functions': 0,
            'num_classes': 0,
            'num_imports': 0,
            'num_variables': 0,
            'num_if_statements': 0,
            'num_for_loops': 0,
            'num_while_loops': 0,
            'num_try_except': 0,
            'max_nesting_depth': 0,
            'num_return_statements': 0,
            'num_print_statements': 0,
            'single_quote_count': 0,
            'double_quote_count': 0,
            'quote_preference': 'none',
            'tab_indented_lines': 0,
            'space_indented_lines': 0,
            'indentation_preference': 'none',
            'snake_case_count': 0,
            'camel_case_count': 0,
            'naming_preference': 'none',
            'comment_lines': 0,
            'comment_ratio': 0,
            'flesch_reading_ease': 0,
            'flesch_kincaid_grade': 0,
            'automated_readability_index': 0,
            'coleman_liau_index': 0,
            'gunning_fog': 0,
            'cyclomatic_complexity': 0,
            'unique_tokens': 0,
            'total_tokens': 0,
            'token_diversity': 0,
            'num_docstrings': 0,
            'total_docstring_length': 0,
            'avg_docstring_length': 0,
            'type_hints_count': 0,
            'return_annotations_count': 0,
            'has_main_guard': False,
            'documentation_ratio': 0
        }
