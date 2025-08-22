import os
import tempfile
import uuid
import threading
import json
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from analyzer import process_videos, allowed_file

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'frame-finder-secret-key'  # In production, use a secure secret key

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# In-memory storage for processing tasks (in production, use Redis or database)
processing_tasks = {}

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

def process_videos_background(task_id, reference_paths, video_paths, frame_interval, confidence_threshold, negative_paths):
    """Process videos in background and update progress"""
    try:
        # Update task status to processing
        processing_tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'results': None,
            'error': None
        }
        
        print(f"Starting background video processing for task {task_id} with {len(video_paths)} videos...")
        
        # Process videos (this will take time)
        results = process_videos(reference_paths, video_paths, frame_interval, confidence_threshold, negative_paths)
        
        # Update task status to completed
        processing_tasks[task_id] = {
            'status': 'completed',
            'progress': 100,
            'results': results,
            'error': None
        }
        
        print(f"Background video processing completed for task {task_id}.")
    except Exception as e:
        # Update task status to error
        processing_tasks[task_id] = {
            'status': 'error',
            'progress': 0,
            'results': None,
            'error': str(e)
        }
        print(f"Error in background video processing for task {task_id}: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and start processing"""
    # Check if files were uploaded
    if 'reference_images' not in request.files or 'videos' not in request.files:
        flash('No files selected')
        return redirect(url_for('index'))
    
    reference_images = request.files.getlist('reference_images')
    negative_references = request.files.getlist('negative_references')
    videos = request.files.getlist('videos')
    
    if not reference_images or not videos:
        flash('Please select both reference images and videos')
        return redirect(url_for('index'))
    
    # Create temporary directories for uploaded files
    temp_dir = tempfile.mkdtemp()
    ref_images_dir = os.path.join(temp_dir, 'reference_images')
    neg_images_dir = os.path.join(temp_dir, 'negative_references')
    videos_dir = os.path.join(temp_dir, 'videos')
    os.makedirs(ref_images_dir, exist_ok=True)
    os.makedirs(neg_images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    # Save reference images
    reference_paths = []
    for ref_image in reference_images:
        if ref_image and allowed_file(ref_image.filename):
            filename = secure_filename(ref_image.filename)
            filepath = os.path.join(ref_images_dir, filename)
            ref_image.save(filepath)
            reference_paths.append(filepath)
    
    # Save negative reference images
    negative_paths = []
    for neg_image in negative_references:
        if neg_image and allowed_file(neg_image.filename):
            filename = secure_filename(neg_image.filename)
            filepath = os.path.join(neg_images_dir, filename)
            neg_image.save(filepath)
            negative_paths.append(filepath)
    
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
    frame_interval = float(request.form.get('frameInterval', 1.0))
    confidence_threshold = float(request.form.get('confidenceThreshold', 75)) / 100.0
    
    # Create a unique task ID
    task_id = str(uuid.uuid4())
    
    # Start background processing
    thread = threading.Thread(
        target=process_videos_background,
        args=(task_id, reference_paths, video_paths, frame_interval, confidence_threshold, negative_paths)
    )
    thread.start()
    
    # Return task ID to client
    return jsonify({'task_id': task_id, 'status': 'started'})

@app.route('/task_status/<task_id>')
def task_status(task_id):
    """Check the status of a processing task"""
    if task_id in processing_tasks:
        return jsonify(processing_tasks[task_id])
    else:
        return jsonify({'status': 'not_found'}), 404

@app.route('/results/<task_id>')
def show_results(task_id):
    """Display results for a completed task"""
    if task_id in processing_tasks:
        task = processing_tasks[task_id]
        if task['status'] == 'completed':
            return render_template('results.html', results=task['results'])
        else:
            flash('Results not available yet. Task status: ' + task['status'])
            return redirect(url_for('index'))
    else:
        flash('Task not found')
        return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True)