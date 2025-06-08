"""
Configuration settings for the AI/ML Code Classifier project.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = True
    
    # File paths
    DATA_DIR = BASE_DIR / 'data'
    MODEL_DIR = BASE_DIR / 'model'
    STATIC_DIR = BASE_DIR / 'static'
    TEMPLATES_DIR = BASE_DIR / 'templates'
    
    # Dataset settings
    TRAIN_DATA_PATH = DATA_DIR / 'training_data.csv'
    TEST_DATA_PATH = DATA_DIR / 'test_data.csv'
    
    # Model settings
    MODEL_PATH = MODEL_DIR / 'code_classifier.joblib'
    VECTORIZER_PATH = MODEL_DIR / 'vectorizer.joblib'
    
    # ML parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_FEATURES = 10000
    
    # Code execution settings
    CODE_TIMEOUT = 10  # seconds
    MAX_OUTPUT_LENGTH = 1000  # characters
    
    # Upload settings
    MAX_FILE_SIZE = 1024 * 1024  # 1MB
    ALLOWED_EXTENSIONS = {'py', 'txt'}

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
