/**
 * AI/ML Code Classifier - Frontend JavaScript
 * Handles user interactions and API communication
 */

$(document).ready(function() {
    // Initialize application
    initializeApp();
    
    // Event listeners
    setupEventListeners();
});

/**
 * Initialize the application
 */
function initializeApp() {
    console.log('🚀 AI/ML Code Classifier initialized');
    
    // Hide results panels initially
    hideAllResults();
    
    // Show welcome message
    showWelcomeMessage();
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // File upload handler
    $('#fileInput').on('change', handleFileUpload);
    
    // Button click handlers
    $('#classifyBtn').on('click', classifyCode);
    $('#executeBtn').on('click', executeCode);
    $('#clearBtn').on('click', clearCode);
    
    // Keyboard shortcuts
    $(document).on('keydown', function(e) {
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case 'Enter':
                    e.preventDefault();
                    classifyCode();
                    break;
                case 'r':
                    e.preventDefault();
                    executeCode();
                    break;
                case 'l':
                    e.preventDefault();
                    clearCode();
                    break;
            }
        }
    });
}

/**
 * Handle file upload
 */
function handleFileUpload(event) {
    const file = event.target.files[0];
    
    if (!file) {
        return;
    }
    
    // Validate file type
    const allowedTypes = ['text/plain', 'text/x-python', 'application/x-python-code'];
    const fileName = file.name.toLowerCase();
    
    if (!fileName.endsWith('.py') && !fileName.endsWith('.txt')) {
        showAlert('Please select a .py or .txt file', 'error');
        return;
    }
    
    // Validate file size (1MB limit)
    if (file.size > 1024 * 1024) {
        showAlert('File size must be less than 1MB', 'error');
        return;
    }
    
    // Create FormData and upload
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading('Uploading file...');
    
    $.ajax({
        url: '/upload',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            if (response.success) {
                // Set the content in the code editor
                if (window.codeEditor) {
                    window.codeEditor.setValue(response.content);
                } else {
                    $('#codeInput').val(response.content);
                }
                
                showAlert(`File "${response.filename}" uploaded successfully!`, 'success');
                hideLoading();
                hideWelcomeMessage();
            } else {
                showAlert(response.error || 'Upload failed', 'error');
                hideLoading();
            }
        },
        error: function(xhr) {
            const response = xhr.responseJSON || {};
            showAlert(response.error || 'Upload failed', 'error');
            hideLoading();
        }
    });
}

/**
 * Classify the code
 */
function classifyCode() {
    const code = getCodeContent();
    
    if (!code.trim()) {
        showAlert('Please enter some Python code to classify', 'warning');
        return;
    }
    
    showLoading('Analyzing code...');
    hideAllResults();
    
    $.ajax({
        url: '/classify',
        type: 'POST',
        data: { code: code },
        success: function(response) {
            hideLoading();
            
            if (response.success) {
                displayClassificationResults(response);
                hideWelcomeMessage();
            } else {
                showAlert(response.error || 'Classification failed', 'error');
            }
        },
        error: function(xhr) {
            hideLoading();
            const response = xhr.responseJSON || {};
            showAlert(response.error || 'Classification failed', 'error');
        }
    });
}

/**
 * Execute the code
 */
function executeCode() {
    const code = getCodeContent();
    
    if (!code.trim()) {
        showAlert('Please enter some Python code to execute', 'warning');
        return;
    }
    
    showLoading('Executing code...');
    hideExecutionResults();
    
    $.ajax({
        url: '/execute',
        type: 'POST',
        data: { code: code },
        success: function(response) {
            hideLoading();
            
            if (response.success) {
                displayExecutionResults(response);
                hideWelcomeMessage();
            } else {
                showAlert(response.error || 'Execution failed', 'error');
            }
        },
        error: function(xhr) {
            hideLoading();
            const response = xhr.responseJSON || {};
            showAlert(response.error || 'Execution failed', 'error');
        }
    });
}

/**
 * Clear the code editor
 */
function clearCode() {
    if (window.codeEditor) {
        window.codeEditor.setValue('');
    } else {
        $('#codeInput').val('');
    }
    
    // Clear file input
    $('#fileInput').val('');
    
    // Hide all results
    hideAllResults();
    showWelcomeMessage();
    
    showAlert('Code cleared', 'info');
}

/**
 * Get code content from editor
 */
function getCodeContent() {
    if (window.codeEditor) {
        return window.codeEditor.getValue();
    } else {
        return $('#codeInput').val();
    }
}

/**
 * Display classification results
 */
function displayClassificationResults(response) {
    // Update prediction
    const prediction = response.prediction;
    const predictionClass = prediction === 'human-written' ? 'prediction-human' : 'prediction-ai';
    $('#predictionResult').text(prediction.toUpperCase()).removeClass().addClass(predictionClass);

    // Update confidence
    const confidence = Math.round(response.confidence * 100);
    let confidenceClass = 'confidence-low';
    if (confidence >= 80) confidenceClass = 'confidence-high';
    else if (confidence >= 60) confidenceClass = 'confidence-medium';

    $('#confidenceResult').text(`${confidence}%`).removeClass().addClass(confidenceClass);

    // Update features
    displayFeatures(response.features);

    // Add model information if available
    if (response.model_name) {
        const modelInfo = `<div class="alert alert-info mt-2">
            <small><i class="fas fa-robot me-1"></i>Classified using: ${response.model_name}</small>
        </div>`;
        $('#featuresAnalysis').append(modelInfo);
    }

    // Add probabilities if available
    if (response.probabilities) {
        const probHtml = `
        <div class="mt-3">
            <h6 class="text-secondary">Prediction Probabilities</h6>
            <div class="progress mb-2">
                <div class="progress-bar bg-success" style="width: ${(response.probabilities['human-written'] * 100).toFixed(1)}%">
                    Human: ${(response.probabilities['human-written'] * 100).toFixed(1)}%
                </div>
            </div>
            <div class="progress">
                <div class="progress-bar bg-danger" style="width: ${(response.probabilities['ai-generated'] * 100).toFixed(1)}%">
                    AI: ${(response.probabilities['ai-generated'] * 100).toFixed(1)}%
                </div>
            </div>
        </div>`;
        $('#featuresAnalysis').append(probHtml);
    }

    // Show results panel
    $('#classificationResults').show().addClass('fade-in');
}

/**
 * Display code features
 */
function displayFeatures(features) {
    const featuresHtml = `
        <div class="row">
            <div class="col-md-6">
                <div class="feature-item">
                    <strong>Lines of Code:</strong> ${features.lines_of_code || 0}
                </div>
                <div class="feature-item">
                    <strong>Character Count:</strong> ${features.character_count || 0}
                </div>
                <div class="feature-item">
                    <strong>Functions:</strong> ${features.num_functions || 0}
                </div>
                <div class="feature-item">
                    <strong>Classes:</strong> ${features.num_classes || 0}
                </div>
            </div>
            <div class="col-md-6">
                <div class="feature-item">
                    <strong>Docstrings:</strong> ${features.num_docstrings || 0}
                </div>
                <div class="feature-item">
                    <strong>Type Hints:</strong> ${features.type_hints_count || 0}
                </div>
                <div class="feature-item">
                    <strong>Complexity:</strong> ${features.complexity_score || 'unknown'}
                </div>
                <div class="feature-item">
                    <strong>Main Guard:</strong> ${features.has_main_guard ? 'Yes' : 'No'}
                </div>
            </div>
        </div>
        ${features.token_diversity ? `
        <div class="feature-item mt-2">
            <strong>Token Diversity:</strong> ${(features.token_diversity * 100).toFixed(1)}%
        </div>
        ` : ''}
    `;

    $('#featuresAnalysis').html(featuresHtml);
}

/**
 * Display execution results
 */
function displayExecutionResults(response) {
    // Update output
    $('#executionOutput').text(response.output || '(no output)');
    
    // Update errors
    $('#executionError').text(response.error || '(no errors)');
    
    // Update return code
    const returnCode = response.return_code;
    const badgeClass = returnCode === 0 ? 'bg-success' : 'bg-danger';
    $('#returnCode').text(returnCode).removeClass().addClass(`badge ${badgeClass}`);
    
    // Show results panel
    $('#executionResults').show().addClass('fade-in');
}

/**
 * Show loading spinner
 */
function showLoading(message = 'Processing...') {
    $('#loadingSpinner p').text(message);
    $('#loadingSpinner').show();
}

/**
 * Hide loading spinner
 */
function hideLoading() {
    $('#loadingSpinner').hide();
}

/**
 * Hide all results panels
 */
function hideAllResults() {
    $('#classificationResults').hide();
    $('#executionResults').hide();
}

/**
 * Hide execution results
 */
function hideExecutionResults() {
    $('#executionResults').hide();
}

/**
 * Show welcome message
 */
function showWelcomeMessage() {
    $('#welcomeMessage').show();
}

/**
 * Hide welcome message
 */
function hideWelcomeMessage() {
    $('#welcomeMessage').hide();
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info') {
    const alertClass = type === 'error' ? 'danger' : type;
    const alertHtml = `
        <div class="alert alert-${alertClass} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // Remove existing alerts
    $('.alert').remove();
    
    // Add new alert
    $('main .container-fluid').prepend(alertHtml);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        $('.alert').fadeOut();
    }, 5000);
}

/**
 * Utility function to format code
 */
function formatCode(code) {
    // Basic code formatting (can be enhanced)
    return code.replace(/\t/g, '    '); // Replace tabs with spaces
}
