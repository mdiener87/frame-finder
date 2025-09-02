# debug_app.py
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
app.secret_key = 'frame-finder-debug-secret-key'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
                            frame_stride=1, resolution_target=1080, lpips_threshold=0.6, clip_threshold=0.2,
                            nms_iou_threshold=0.5, debounce_n=2, debounce_m=8):
    """Process videos in background and update progress with debug info"""
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
            'cancelled': False,
            'debug_info': []  # Add debug info
        }
        
        print(f"[DEBUG] Starting background video processing for task {task_id} with {len(video_paths)} videos...")
        
        # Add debug info
        debug_info = []
        debug_info.append(f"Task ID: {task_id}")
        debug_info.append(f"Number of reference images: {len(reference_paths)}")
        debug_info.append(f"Number of videos: {len(video_paths)}")
        debug_info.append(f"Frame interval: {frame_interval}")
        debug_info.append(f"LPIPS threshold: {lpips_threshold}")
        debug_info.append(f"CLIP threshold: {clip_threshold}")
        processing_tasks[task_id]['debug_info'] = debug_info
        
        # Define progress callback function
        def progress_callback(progress_info):
            if task_id in processing_tasks and not processing_tasks[task_id].get('cancelled', False):
                # Calculate overall progress based on videos and frames
                if progress_info['status'] == 'processing_video':
                    # When starting a new video, update current video name
                    processing_tasks[task_id]['current_video'] = progress_info['current_video']
                    processing_tasks[task_id]['debug_info'].append(f"Processing video: {progress_info['current_video']}")
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
        print("[DEBUG] Calling process_videos...")
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
        
        print(f"[DEBUG] process_videos completed. Results type: {type(results)}")
        print(f"[DEBUG] Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
        
        # Add debug info about results
        if isinstance(results, dict):
            for video_name, video_data in results.items():
                if isinstance(video_data, dict) and 'matches' in video_data:
                    matches = video_data['matches']
                    debug_info.append(f"Video '{video_name}' has {len(matches)} matches")
                    print(f"[DEBUG] Video '{video_name}' has {len(matches)} matches")
                    for i, match in enumerate(matches[:3]):  # Show first 3 matches
                        debug_info.append(f"  Match {i+1}: timestamp={match.get('timestamp', 'N/A')}, confidence={match.get('confidence', 'N/A')}")
                        print(f"[DEBUG]   Match {i+1}: timestamp={match.get('timestamp', 'N/A')}, confidence={match.get('confidence', 'N/A')}")
                else:
                    debug_info.append(f"Video '{video_name}' data structure: {type(video_data)}")
                    print(f"[DEBUG] Video '{video_name}' data structure: {type(video_data)}")
        
        # Check if task was cancelled during processing
        if task_id in processing_tasks and processing_tasks[task_id].get('cancelled', False):
            print(f"[DEBUG] Background video processing cancelled for task {task_id}.")
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
            'cancelled': False,
            'debug_info': debug_info
        }
        
        print(f"[DEBUG] Background video processing completed for task {task_id}.")
    except Exception as e:
        # Update task status to error
        processing_tasks[task_id] = {
            'status': 'error',
            'progress': 0,
            'results': None,
            'error': str(e),
            'debug_info': [f"Error: {str(e)}"]
        }
        print(f"[DEBUG] Error in background video processing for task {task_id}: {str(e)}")
        import traceback
        traceback.print_exc()

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and start processing with debug info"""
    print("[DEBUG] Upload endpoint called")
    
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
    
    print(f"[DEBUG] Files received - References: {len(reference_images)}, Negatives: {len(negative_references)}, Videos: {len(videos)}, Directory files: {len(video_directory_files)}")
    
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
            print(f"[DEBUG] Saved reference image: {filepath}")
    
    # Save negative reference images
    negative_paths = []
    for neg_image in negative_references:
        if neg_image and allowed_file(neg_image.filename):
            filename = secure_filename(neg_image.filename)
            filepath = os.path.join(neg_images_dir, filename)
            neg_image.save(filepath)
            negative_paths.append(filepath)
            print(f"[DEBUG] Saved negative reference image: {filepath}")
    
    # Save videos (both individual files and directory files)
    video_paths = []
    
    # Process individual video files
    for video in videos:
        if video and allowed_file(video.filename):
            filename = secure_filename(video.filename)
            filepath = os.path.join(videos_dir, filename)
            video.save(filepath)
            video_paths.append(filepath)
            print(f"[DEBUG] Saved video file: {filepath}")
    
    # Process video directory files
    for video in video_directory_files:
        if video and allowed_file(video.filename):
            # For directory uploads, we need to preserve the directory structure
            # or just use the filename if webkitRelativePath is not available
            filename = secure_filename(video.filename)
            filepath = os.path.join(videos_dir, filename)
            video.save(filepath)
            video_paths.append(filepath)
            print(f"[DEBUG] Saved directory video file: {filepath}")
    
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
    # Use a default confidence threshold of 75% since we removed the slider
    confidence_threshold = 0.75
    
    print(f"[DEBUG] Processing parameters:")
    print(f"  Frame interval: {frame_interval}")
    print(f"  Frame stride: {frame_stride}")
    print(f"  Resolution target: {resolution_target}")
    print(f"  LPIPS threshold: {lpips_threshold}")
    print(f"  CLIP threshold: {clip_threshold}")
    print(f"  NMS IoU threshold: {nms_iou_threshold}")
    print(f"  Debounce N: {debounce_n}")
    print(f"  Debounce M: {debounce_m}")
    
    # Create a unique task ID
    task_id = str(uuid.uuid4())
    print(f"[DEBUG] Created task ID: {task_id}")
    
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
    print(f"[DEBUG] Task status requested for task_id: {task_id}")
    if task_id in processing_tasks:
        task_info = processing_tasks[task_id]
        print(f"[DEBUG] Task status: {task_info.get('status', 'unknown')}")
        return jsonify(task_info)
    else:
        print(f"[DEBUG] Task not found: {task_id}")
        return jsonify({'status': 'not_found'}), 404

@app.route('/results/<task_id>')
def show_results(task_id):
    """Display results for a completed task"""
    print(f"[DEBUG] Results requested for task_id: {task_id}")
    if task_id in processing_tasks:
        task = processing_tasks[task_id]
        print(f"[DEBUG] Task status: {task.get('status', 'unknown')}")
        if task['status'] == 'completed':
            results = task.get('results', {})
            print(f"[DEBUG] Results type: {type(results)}")
            if isinstance(results, dict):
                print(f"[DEBUG] Results keys: {list(results.keys())}")
                for video_name, video_data in results.items():
                    if isinstance(video_data, dict) and 'matches' in video_data:
                        matches = video_data['matches']
                        print(f"[DEBUG] Video '{video_name}' has {len(matches)} matches")
                    else:
                        print(f"[DEBUG] Video '{video_name}' data structure: {type(video_data)}")
            
            return render_template('results.html', results=results, task_id=task_id)
        else:
            flash('Results not available yet. Task status: ' + task['status'])
            return redirect(url_for('index'))
    else:
        flash('Task not found')
        return redirect(url_for('index'))

@app.route('/debug_info/<task_id>')
def debug_info(task_id):
    """Display debug information for a task"""
    if task_id in processing_tasks:
        task = processing_tasks[task_id]
        debug_info = task.get('debug_info', [])
        return '<br>'.join(debug_info)
    else:
        return 'Task not found', 404

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use a different port to avoid conflicts