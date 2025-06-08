# AI/ML Code Classifier

A supervised machine learning system that classifies Python code as either AI-generated or human-written, with a web interface for easy interaction.

## 🎯 Project Overview

This project implements a text classification system using machine learning to distinguish between AI-generated and human-written Python code. It includes:

- **Machine Learning Model**: Supervised learning classifier using NLP techniques
- **Web Interface**: Flask-based web application for code input and classification
- **Code Execution**: Safe execution environment for testing Python code
- **Real-time Analysis**: Instant classification with confidence scores

## 🚀 Features

- ✅ Upload or paste Python code for classification
- ✅ Real-time AI vs Human classification with confidence scores
- ✅ Safe code execution within the application
- ✅ Detailed analysis of code characteristics
- ✅ Responsive web interface
- ✅ Model training and evaluation tools

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, transformers
- **Frontend**: HTML/CSS/JavaScript, Bootstrap
- **Data Processing**: pandas, numpy, nltk
- **Code Analysis**: AST parsing, textstat

## 📋 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AIML
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open browser**: Navigate to `http://localhost:5000`

## 📊 Project Structure

```
AIML/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── requirements.txt       # Dependencies
├── data/                 # Training datasets
├── model/                # Trained models
├── static/               # CSS, JS, images
├── templates/            # HTML templates
├── utils/                # Helper functions
└── tests/                # Unit tests
```

## 🎓 Learning Objectives

This project demonstrates:
- Supervised machine learning for text classification
- Natural Language Processing (NLP) techniques
- Web application development with Flask
- Data preprocessing and feature engineering
- Model evaluation and validation
- Safe code execution practices

## 📈 Development Phases

1. **Phase 1**: Project setup and environment
2. **Phase 2**: Data collection and processing
3. **Phase 3**: Machine learning model development
4. **Phase 4**: Web interface implementation
5. **Phase 5**: Code execution and final integration

## 🤝 Contributing

This is an educational project. Feel free to fork and experiment!

## 📄 License

MIT License - see LICENSE file for details.
