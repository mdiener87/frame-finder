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

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Removed MAX_CONTENT_LENGTH to allow unlimited upload size for localhost

# In-memory storage for processing tasks (in production, use Redis or database)
processing_tasks = {}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Custom filter to extract basename from path
@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path)

# Custom filter to format seconds to hh:mm:ss
@app.template_filter('format_time')
def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"

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
    if 'reference_images' not in request.files:
        flash('No reference images selected')
        return redirect(url_for('index'))
    
    reference_images = request.files.getlist('reference_images')
    negative_references = request.files.getlist('negative_references')
    videos = request.files.getlist('videos')
    video_directory_files = request.files.getlist('videoDirectory')
    
    if not reference_images:
        flash('Please select reference images')
        return redirect(url_for('index'))
    
    # Check if either individual videos or directory videos were selected
    if not videos and not video_directory_files:
        flash('Please select either video files or a video directory')
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
    
    # Save videos (both individual files and directory files)
    video_paths = []
    
    # Process individual video files
    for video in videos:
        if video and allowed_file(video.filename):
            filename = secure_filename(video.filename)
            filepath = os.path.join(videos_dir, filename)
            video.save(filepath)
            video_paths.append(filepath)
    
    # Process video directory files
    for video in video_directory_files:
        if video and allowed_file(video.filename):
            # For directory uploads, we need to preserve the directory structure
            # or just use the filename if webkitRelativePath is not available
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