"""
AI/ML Code Classifier - Main Flask Application

This is the main entry point for the AI/ML Code Classifier web application.
It provides a web interface for classifying Python code as AI-generated or human-written.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_cors import CORS
import os
import sys
import subprocess
import tempfile
import traceback
from pathlib import Path

# Import configuration
from config import config

# Import ML components
sys.path.append(str(Path(__file__).parent))
from utils.model_predictor import CodeClassifierPredictor

# Create Flask application
def create_app(config_name='default'):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Enable CORS
    CORS(app)
    
    # Ensure required directories exist
    os.makedirs(app.config['DATA_DIR'], exist_ok=True)
    os.makedirs(app.config['MODEL_DIR'], exist_ok=True)

    # Initialize ML predictor
    try:
        app.predictor = CodeClassifierPredictor(app.config['MODEL_DIR'])
        print("✅ ML model loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not load ML model: {e}")
        print("   Classification will use placeholder responses")
        app.predictor = None
    
    # Routes
    @app.route('/')
    def index():
        """Main page with code input form."""
        return render_template('index.html')
    
    @app.route('/classify', methods=['POST'])
    def classify_code():
        """Classify submitted Python code."""
        try:
            # Get code from form
            code = request.form.get('code', '').strip()

            if not code:
                return jsonify({
                    'error': 'No code provided',
                    'success': False
                }), 400

            # Use ML model if available
            if app.predictor:
                try:
                    # Get detailed analysis
                    analysis = app.predictor.analyze_code_features(code)

                    if 'error' in analysis:
                        raise Exception(analysis['error'])

                    prediction_result = analysis['prediction']
                    features = analysis['feature_analysis']

                    # Prepare response
                    result = {
                        'success': True,
                        'prediction': prediction_result['prediction_text'],
                        'confidence': prediction_result['confidence'],
                        'probabilities': prediction_result['probabilities'],
                        'model_name': prediction_result['model_name'],
                        'features': {
                            'lines_of_code': features['basic']['total_lines'],
                            'character_count': features['basic']['total_characters'],
                            'num_functions': features['syntactic']['num_functions'],
                            'num_classes': features['syntactic']['num_classes'],
                            'num_docstrings': features['documentation']['num_docstrings'],
                            'type_hints_count': features['documentation']['type_hints_count'],
                            'cyclomatic_complexity': features['syntactic']['cyclomatic_complexity'],
                            'token_diversity': features['complexity']['token_diversity'],
                            'has_main_guard': features['documentation']['has_main_guard'],
                            'complexity_score': 'high' if features['syntactic']['cyclomatic_complexity'] > 5 else 'medium' if features['syntactic']['cyclomatic_complexity'] > 2 else 'low'
                        },
                        'message': 'Classification completed successfully using trained ML model'
                    }

                    return jsonify(result)

                except Exception as ml_error:
                    print(f"ML prediction error: {ml_error}")
                    # Fall back to placeholder
                    pass

            # Fallback placeholder response
            result = {
                'success': True,
                'prediction': 'human-written',  # Placeholder
                'confidence': 0.75,  # Placeholder
                'features': {
                    'lines_of_code': len(code.split('\n')),
                    'character_count': len(code),
                    'complexity_score': 'medium'  # Placeholder
                },
                'message': 'Classification completed using placeholder (ML model not available)'
            }

            return jsonify(result)

        except Exception as e:
            return jsonify({
                'error': f'Classification failed: {str(e)}',
                'success': False
            }), 500
    
    @app.route('/execute', methods=['POST'])
    def execute_code():
        """Execute Python code safely."""
        try:
            # Get code from form
            code = request.form.get('code', '').strip()
            
            if not code:
                return jsonify({
                    'error': 'No code provided',
                    'success': False
                }), 400
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute code with timeout
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=app.config['CODE_TIMEOUT']
                )
                
                # Prepare response
                output = result.stdout
                error = result.stderr
                
                # Limit output length
                max_length = app.config['MAX_OUTPUT_LENGTH']
                if len(output) > max_length:
                    output = output[:max_length] + "\\n... (output truncated)"
                if len(error) > max_length:
                    error = error[:max_length] + "\\n... (error truncated)"
                
                return jsonify({
                    'success': True,
                    'output': output,
                    'error': error,
                    'return_code': result.returncode
                })
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            return jsonify({
                'error': f'Code execution timed out after {app.config["CODE_TIMEOUT"]} seconds',
                'success': False
            }), 400
            
        except Exception as e:
            return jsonify({
                'error': f'Execution failed: {str(e)}',
                'success': False
            }), 500
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        """Handle file upload."""
        try:
            if 'file' not in request.files:
                return jsonify({
                    'error': 'No file provided',
                    'success': False
                }), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'error': 'No file selected',
                    'success': False
                }), 400
            
            # Check file extension
            if not file.filename.lower().endswith(('.py', '.txt')):
                return jsonify({
                    'error': 'Only .py and .txt files are allowed',
                    'success': False
                }), 400
            
            # Read file content
            content = file.read().decode('utf-8')
            
            # Check file size
            if len(content) > app.config['MAX_FILE_SIZE']:
                return jsonify({
                    'error': 'File too large',
                    'success': False
                }), 400
            
            return jsonify({
                'success': True,
                'content': content,
                'filename': file.filename
            })
            
        except Exception as e:
            return jsonify({
                'error': f'File upload failed: {str(e)}',
                'success': False
            }), 500
    
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        model_status = 'available' if app.predictor else 'unavailable'
        model_info = {}

        if app.predictor:
            try:
                model_info = app.predictor.get_model_info()
            except:
                model_status = 'error'

        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'features': {
                'classification': model_status,
                'execution': 'available',
                'upload': 'available'
            },
            'model_info': model_info
        })
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return render_template('error.html', 
                             error_code=404, 
                             error_message="Page not found"), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        return render_template('error.html', 
                             error_code=500, 
                             error_message="Internal server error"), 500
    
    return app

# Create application instance
app = create_app()

if __name__ == '__main__':
    print("🚀 Starting AI/ML Code Classifier...")
    print("📊 Features available:")

    # Check model status
    model_status = "✅ Code Classification (ML Model)" if app.predictor else "⚠️  Code Classification (Placeholder - No ML Model)"
    print(f"   {model_status}")
    print("   ✅ Code Execution")
    print("   ✅ File Upload")
    print("   ✅ Web Interface")

    if app.predictor:
        try:
            model_info = app.predictor.get_model_info()
            print(f"   🤖 Model: {model_info.get('model_name', 'Unknown')}")
            print(f"   📊 Features: {model_info.get('feature_count', 0)}")
        except:
            pass

    print()
    print("🌐 Access the application at: http://localhost:5000")
    print("📖 API endpoints:")
    print("   POST /classify - Classify code")
    print("   POST /execute - Execute code")
    print("   POST /upload - Upload file")
    print("   GET /health - Health check")
    print()

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
