import os
import tempfile
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from analyzer import process_videos, allowed_file
import json

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'frame-finder-secret-key'  # In production, use a secure secret key

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Custom filter to extract basename from path
@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path)

@app.route('/')
def index():
    """Render the main upload page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and start processing"""
    # Check if files were uploaded
    if 'reference_images' not in request.files or 'videos' not in request.files:
        flash('No files selected')
        return redirect(url_for('index'))
    
    reference_images = request.files.getlist('reference_images')
    videos = request.files.getlist('videos')
    
    if not reference_images or not videos:
        flash('Please select both reference images and videos')
        return redirect(url_for('index'))
    
    # Create temporary directories for uploaded files
    temp_dir = tempfile.mkdtemp()
    ref_images_dir = os.path.join(temp_dir, 'reference_images')
    videos_dir = os.path.join(temp_dir, 'videos')
    os.makedirs(ref_images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    # Save reference images
    reference_paths = []
    for ref_image in reference_images:
        if ref_image and allowed_file(ref_image.filename):
            filename = secure_filename(ref_image.filename)
            filepath = os.path.join(ref_images_dir, filename)
            ref_image.save(filepath)
            reference_paths.append(filepath)
    
    # Save videos
    video_paths = []
    for video in videos:
        if video and allowed_file(video.filename):
            filename = secure_filename(video.filename)
            filepath = os.path.join(videos_dir, filename)
            video.save(filepath)
            video_paths.append(filepath)
    
    if not reference_paths or not video_paths:
        flash('No valid files uploaded')
        return redirect(url_for('index'))
    
    # Get processing parameters from form
    frame_interval = int(request.form.get('frameInterval', 1))
    confidence_threshold = float(request.form.get('confidenceThreshold', 50)) / 100.0
    
    # Process videos
    try:
        results = process_videos(reference_paths, video_paths, frame_interval, confidence_threshold)
        return render_template('results.html', results=results)
    except Exception as e:
        flash(f'Error processing videos: {str(e)}')
        return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True)