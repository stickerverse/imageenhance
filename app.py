import os
import uuid
import time
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch

# Import the Neural Enhancement Pipeline
from neural_enhancement_pipeline import NeuralEnhancementPipeline

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default-dev-key-change-in-production')

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = Path('uploads')
RESULTS_FOLDER = Path('results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Create folders if they don't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

# Set upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize the Neural Enhancement Pipeline
# Check for CUDA availability but default to CPU for production reliability
use_cuda = torch.cuda.is_available()
enhancement_pipeline = NeuralEnhancementPipeline(use_cuda=use_cuda)

# Image type options for the UI
IMAGE_TYPES = {
    'portrait': 'Portrait (People)',
    'landscape': 'Landscape',
    'food': 'Food',
    'night': 'Night Scene',
    'macro': 'Macro',
    'document': 'Document',
    'black_and_white': 'Black & White'
}

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', image_types=IMAGE_TYPES)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and initial processing"""
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user does not select file
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        unique_id = str(uuid.uuid4())
        original_extension = os.path.splitext(file.filename)[1]
        filename = unique_id + original_extension
        
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Redirect to the enhancement page
        return redirect(url_for('enhance', filename=filename))
    
    flash('Unsupported file type. Please upload JPG, PNG, or WEBP images.')
    return redirect(url_for('index'))

@app.route('/enhance/<filename>', methods=['GET', 'POST'])
def enhance(filename):
    """Handle image enhancement with options"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        flash('File not found')
        return redirect(url_for('index'))
    
    # If POST, process the image
    if request.method == 'POST':
        # Get enhancement parameters from the form
        image_type = request.form.get('image_type', 'portrait')
        strength = float(request.form.get('strength', 0.5))
        
        # Create a unique ID for the enhanced image
        unique_id = str(uuid.uuid4())
        result_filename = unique_id + os.path.splitext(filename)[1]
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # Enhancement parameters
        enhancement_params = {
            'detail_strength': strength,
            'color_strength': strength,
            'frequency_strength': strength
        }
        
        try:
            # Load image
            input_image = Image.open(file_path)
            
            # Process image
            start_time = time.time()
            enhanced_image = enhancement_pipeline.enhance_image(
                input_image, 
                image_type, 
                enhancement_params
            )
            processing_time = time.time() - start_time
            
            # Save the enhanced image
            enhanced_image.save(result_path)
            
            # Redirect to the results page
            return redirect(url_for('result', filename=result_filename, original=filename, processing_time=processing_time))
        
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('enhance', filename=filename))
    
    # Display the enhancement options page
    return render_template('enhance.html', filename=filename, image_types=IMAGE_TYPES)

@app.route('/result/<filename>')
def result(filename):
    """Display the enhanced image and download option"""
    original = request.args.get('original', '')
    processing_time = request.args.get('processing_time', '')
    
    # Try to format processing time as a float
    try:
        processing_time = f"{float(processing_time):.2f} seconds"
    except (ValueError, TypeError):
        processing_time = "Unknown"
    
    return render_template('result.html', 
                           filename=filename, 
                           original=original,
                           processing_time=processing_time)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    """Download the enhanced image"""
    return send_from_directory(app.config['RESULTS_FOLDER'], 
                               filename, 
                               as_attachment=True)

@app.route('/api/enhance', methods=['POST'])
def api_enhance():
    """API endpoint for enhancement"""
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Get parameters
        image_type = request.form.get('image_type', 'portrait')
        strength = float(request.form.get('strength', 0.5))
        
        # Generate unique filenames
        unique_id = str(uuid.uuid4())
        original_extension = os.path.splitext(file.filename)[1]
        filename = unique_id + original_extension
        result_filename = "enhanced_" + filename
        
        # Save paths
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # Save the uploaded file
        file.save(file_path)
        
        # Enhancement parameters
        enhancement_params = {
            'detail_strength': strength,
            'color_strength': strength,
            'frequency_strength': strength
        }
        
        try:
            # Load image
            input_image = Image.open(file_path)
            
            # Process image
            start_time = time.time()
            enhanced_image = enhancement_pipeline.enhance_image(
                input_image, 
                image_type, 
                enhancement_params
            )
            processing_time = time.time() - start_time
            
            # Save the enhanced image
            enhanced_image.save(result_path)
            
            # Return the download URL
            return jsonify({
                'result_url': url_for('result_file', filename=result_filename, _external=True),
                'download_url': url_for('download_file', filename=result_filename, _external=True),
                'processing_time': processing_time
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Unsupported file type. Please upload JPG, PNG, or WEBP images.'}), 400

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    flash('File too large. Maximum size is 16MB.')
    return redirect(url_for('index')), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server error"""
    flash('Server error occurred. Please try again later.')
    return redirect(url_for('index')), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    # Use host 0.0.0.0 to make the app accessible externally
    app.run(host='0.0.0.0', port=port, debug=False)
