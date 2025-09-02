import os
import tempfile
import uuid
import threading
import json
import datetime
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

def process_videos_background(task_id, reference_paths, video_paths, frame_interval, confidence_threshold, negative_paths,
                            frame_stride=1, resolution_target=1080, lpips_threshold=0.35, clip_threshold=0.33,
                            nms_iou_threshold=0.5, debounce_n=3, debounce_m=12):
    """Process videos in background and update progress"""
    try:
        # Update task status to processing
        processing_tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'current_video': '',
            'results': None,
            'reference_paths': reference_paths,
            'negative_paths': negative_paths,
            'frame_interval': frame_interval,
            'confidence_threshold': confidence_threshold,
            'frame_stride': frame_stride,
            'resolution_target': resolution_target,
            'lpips_threshold': lpips_threshold,
            'clip_threshold': clip_threshold,
            'nms_iou_threshold': nms_iou_threshold,
            'debounce_n': debounce_n,
            'debounce_m': debounce_m,
            'error': None,
            'cancelled': False  # Add cancellation flag
        }
        
        print(f"Starting background video processing for task {task_id} with {len(video_paths)} videos...")
        
        # Define progress callback function
        def progress_callback(progress_info):
            if task_id in processing_tasks and not processing_tasks[task_id].get('cancelled', False):
                # Calculate overall progress based on videos and frames
                if progress_info['status'] == 'processing_video':
                    # When starting a new video, update current video name
                    processing_tasks[task_id]['current_video'] = progress_info['current_video']
                    # Calculate progress based on video index
                    video_progress = (progress_info['video_index'] / progress_info['total_videos']) * 100
                    processing_tasks[task_id]['progress'] = max(0, min(100, video_progress))
                elif progress_info['status'] == 'processing_frames':
                    # Calculate progress within the current video based on frames processed
                    video_progress = (progress_info['video_index'] / progress_info['total_videos']) * 100
                    if progress_info['total_frames'] > 0:
                        frame_progress = (progress_info['current_frame'] / progress_info['total_frames']) * (1 / progress_info['total_videos']) * 100
                        total_progress = video_progress + frame_progress
                        processing_tasks[task_id]['progress'] = max(0, min(100, total_progress))
        
        # Process videos (this will take time) with progress callback
        results = process_videos(
            reference_paths, video_paths, negative_paths,
            frame_interval=frame_interval,
            frame_stride=frame_stride,
            resolution_target=resolution_target,
            lpips_threshold=lpips_threshold,
            clip_threshold=clip_threshold,
            nms_iou_threshold=nms_iou_threshold,
            debounce_n=debounce_n,
            debounce_m=debounce_m,
            progress_callback=progress_callback
        )
        
        # Check if task was cancelled during processing
        if task_id in processing_tasks and processing_tasks[task_id].get('cancelled', False):
            print(f"Background video processing cancelled for task {task_id}.")
            return
        
        # Update task status to completed
        processing_tasks[task_id] = {
            'status': 'completed',
            'progress': 100,
            'results': results,
            'reference_paths': reference_paths,
            'negative_paths': negative_paths,
            'frame_interval': frame_interval,
            'confidence_threshold': confidence_threshold,
            'error': None,
            'cancelled': False
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
        error_msg = 'No reference images selected'
        # Check if this is an AJAX request
        if request.headers.get('Content-Type') == 'application/json' or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': error_msg}), 400
        else:
            flash(error_msg)
            return redirect(url_for('index'))
    
    reference_images = request.files.getlist('reference_images')
    negative_references = request.files.getlist('negative_references')
    videos = request.files.getlist('videos')
    video_directory_files = request.files.getlist('videoDirectory')
    
    if not reference_images:
        error_msg = 'Please select reference images'
        # Check if this is an AJAX request
        if request.headers.get('Content-Type') == 'application/json' or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': error_msg}), 400
        else:
            flash(error_msg)
            return redirect(url_for('index'))
    
    # Check if either individual videos or directory videos were selected
    if not videos and not video_directory_files:
        error_msg = 'Please select either video files or a video directory'
        # Check if this is an AJAX request
        if request.headers.get('Content-Type') == 'application/json' or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': error_msg}), 400
        else:
            flash(error_msg)
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
        error_msg = 'No valid files uploaded'
        # Check if this is an AJAX request
        if request.headers.get('Content-Type') == 'application/json' or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': error_msg}), 400
        else:
            flash(error_msg)
            return redirect(url_for('index'))
    
    # Get processing parameters from form
    frame_interval = float(request.form.get('frameInterval', 1.0))
    frame_stride = int(request.form.get('frameStride', 1))
    resolution_target = int(request.form.get('resolutionTarget', 1080))
    lpips_threshold = float(request.form.get('lpipsThreshold', 0.6))  # More reasonable default
    clip_threshold = float(request.form.get('clipThreshold', 0.2))   # More reasonable default
    nms_iou_threshold = float(request.form.get('nmsThreshold', 0.5))
    debounce_n = int(request.form.get('debounceN', 2))  # More reasonable default
    debounce_m = int(request.form.get('debounceM', 8))  # More reasonable default
    # Get confidence threshold from form (default to 0 for showing all matches)
    confidence_threshold = float(request.form.get('confidenceThreshold', 0.0))
    
    # Create a unique task ID
    task_id = str(uuid.uuid4())
    
    # Start background processing
    thread = threading.Thread(
        target=process_videos_background,
        args=(task_id, reference_paths, video_paths, frame_interval, confidence_threshold, negative_paths),
        kwargs={
            'frame_stride': frame_stride,
            'resolution_target': resolution_target,
            'lpips_threshold': lpips_threshold,
            'clip_threshold': clip_threshold,
            'nms_iou_threshold': nms_iou_threshold,
            'debounce_n': debounce_n,
            'debounce_m': debounce_m
        }
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
            return render_template('results.html', results=task['results'], task_id=task_id)
        else:
            flash('Results not available yet. Task status: ' + task['status'])
            return redirect(url_for('index'))
    else:
        flash('Task not found')
        return redirect(url_for('index'))

@app.route('/cancel_task/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """Cancel a processing task"""
    if task_id in processing_tasks:
        # Set cancelled flag
        processing_tasks[task_id]['cancelled'] = True
        processing_tasks[task_id]['status'] = 'cancelled'
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'error': 'Task not found'}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/export_results/<task_id>', methods=['POST'])
def export_results(task_id):
    """Export results as JSON based on current confidence filters"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = processing_tasks[task_id]
    if task['status'] != 'completed':
        return jsonify({'error': 'Task not completed yet'}), 400
    
    # Get confidence thresholds from request
    request_data = request.get_json()
    confidence_thresholds = request_data.get('confidence_thresholds', {})
    
    # Get other task information
    reference_paths = task.get('reference_paths', [])
    negative_paths = task.get('negative_paths', [])
    results = task.get('results', {})
    
    # Format data according to our JSON schema
    export_data = {
        'frameFinderGuid': task_id,
        'date': datetime.datetime.utcnow().isoformat() + 'Z',
        'minConfidenceLevel': min(confidence_thresholds.values()) if confidence_thresholds else 0.0,
        'referenceImages': [os.path.basename(path) for path in reference_paths],
        'negativeImages': [os.path.basename(path) for path in negative_paths],
        'analyzedVideos': list(results.keys()),
        'analysisResults': []
    }
    
    # Process each video's results
    for video_name, video_data in results.items():
        matches = video_data.get('matches', [])
        max_confidence = video_data.get('max_confidence', 0.0)
        
        # Get confidence threshold for this video (default to 0 if not provided)
        confidence_threshold = confidence_thresholds.get(video_name, 0.0) / 100.0
        
        # Filter matches based on confidence threshold
        filtered_matches = [match for match in matches if match.get('confidence', 0.0) >= confidence_threshold]
        
        # Sort matches by confidence (descending)
        filtered_matches.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        # Format time indices as hh:mm:ss.SSS
        def format_time(seconds):
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"
        
        # Create topFrames array
        top_frames = [
            {
                'timeIndex': format_time(match.get('timestamp', 0.0)),
                'confidence': match.get('confidence', 0.0)
            }
            for match in filtered_matches
        ]
        
        # Add to analysis results
        export_data['analysisResults'].append({
            'fileName': video_name,
            'maxConfidence': max_confidence,
            'topFrames': top_frames
        })
    
    # Return JSON data with appropriate headers for file download
    response = jsonify(export_data)
    response.headers['Content-Disposition'] = f'attachment; filename=frame_finder_export_{task_id}.json'
    return response

if __name__ == '__main__':
    app.run(debug=True)