import os
import cv2
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames(video_path, interval=1):
    """Extract frames from video at specified interval (in seconds)
    
    Args:
        video_path (str): Path to the video file
        interval (int): Interval in seconds between frames (default: 1)
        
    Returns:
        list: List of tuples (frame, timestamp) where frame is a PIL Image
    """
    frames = []
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    
    frame_count = 0
    timestamp = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append((pil_image, timestamp))
        
        frame_count += 1
        timestamp = frame_count / fps
    
    cap.release()
    return frames

def compare_images(reference_images, frame):
    """Compare a frame against reference images using CLIP similarity
    
    Args:
        reference_images (list): List of reference PIL Images
        frame (PIL.Image): Frame to compare against references
        
    Returns:
        list: List of tuples (reference_index, similarity_score)
    """
    similarities = []
    
    # Process the frame and reference images
    frame_inputs = processor(images=frame, return_tensors="pt", padding=True)
    
    # Get frame embedding
    with torch.no_grad():
        frame_features = model.get_image_features(**frame_inputs)
        frame_features = frame_features / frame_features.norm(p=2, dim=-1, keepdim=True)
    
    # Compare with each reference image
    for i, ref_img in enumerate(reference_images):
        ref_inputs = processor(images=ref_img, return_tensors="pt", padding=True)
        
        # Get reference image embedding
        with torch.no_grad():
            ref_features = model.get_image_features(**ref_inputs)
            ref_features = ref_features / ref_features.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(frame_features, ref_features).item()
        similarities.append((i, similarity))
    
    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

def process_videos(reference_paths, video_paths, frame_interval=1, confidence_threshold=0.5):
    """Process videos and find matches with reference images
    
    Args:
        reference_paths (list): List of paths to reference images
        video_paths (list): List of paths to video files
        frame_interval (int): Interval in seconds between frames (default: 1)
        confidence_threshold (float): Minimum similarity score to consider a match (default: 0.5)
        
    Returns:
        dict: Results organized by video with list of matches
    """
    # Load reference images
    reference_images = []
    for path in reference_paths:
        ref_img = Image.open(path)
        reference_images.append(ref_img)
    
    results = {}
    
    # Process each video
    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        results[video_name] = []
        
        try:
            # Extract frames
            frames = extract_frames(video_path, frame_interval)
            
            # Compare each frame with reference images
            for frame, timestamp in frames:
                similarities = compare_images(reference_images, frame)
                
                # Check if any similarity exceeds threshold
                for ref_index, similarity in similarities:
                    if similarity >= confidence_threshold:
                        # Add match to results
                        match = {
                            'timestamp': timestamp,
                            'confidence': similarity,
                            'reference_image': reference_paths[ref_index]
                        }
                        results[video_name].append(match)
                        
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            results[video_name].append({
                'error': str(e)
            })
    
    return results

# Additional utility functions can be added here as needed