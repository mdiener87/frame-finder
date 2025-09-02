# debug_app_results.py
import json
import os
from analyzer import process_videos
import tempfile
import numpy as np
from PIL import Image
import cv2

def create_debug_test():
    """Create a simple test to debug app results"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create a simple red square reference image (50x50)
    ref_image = Image.new('RGB', (50, 50), color='red')
    ref_path = os.path.join(temp_dir, 'reference.png')
    ref_image.save(ref_path)
    
    # Create a simple test video with the red square appearing in some frames
    video_path = os.path.join(temp_dir, 'test_video.mp4')
    
    # Video parameters
    width, height = 320, 240
    fps = 10
    duration = 3  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Generate frames
    for i in range(total_frames):
        # Create a white background frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # In some frames, add the red square
        if i in [10, 15, 20]:  # Add square in these frames
            # Place red square at different positions
            x_pos = 100 + (i % 5) * 20
            y_pos = 80 + (i % 3) * 30
            # Create a red square (BGR format for OpenCV)
            frame[y_pos:y_pos+50, x_pos:x_pos+50] = [0, 0, 255]
        
        out.write(frame)
    
    out.release()
    
    return [ref_path], [video_path]

def debug_app_processing():
    """Debug the app processing flow"""
    print("Creating debug test data...")
    ref_paths, video_paths = create_debug_test()
    
    print(f"Reference paths: {ref_paths}")
    print(f"Video paths: {video_paths}")
    
    # Process with the same parameters as the app would use
    print("Processing with analyzer...")
    results = process_videos(
        reference_paths=ref_paths,
        video_paths=video_paths,
        frame_interval=1.0,
        frame_stride=1,
        resolution_target=1080,
        lpips_threshold=0.6,  # Same as app defaults
        clip_threshold=0.2,   # Same as app defaults
        nms_iou_threshold=0.5,
        debounce_n=2,
        debounce_m=8
    )
    
    print("Results from analyzer:")
    print(json.dumps(results, indent=2, default=str))
    
    # Check the structure
    for video_name, video_data in results.items():
        print(f"\nVideo: {video_name}")
        print(f"  Type: {type(video_data)}")
        if isinstance(video_data, dict):
            print(f"  Keys: {list(video_data.keys())}")
            if 'matches' in video_data:
                matches = video_data['matches']
                print(f"  Number of matches: {len(matches)}")
                for i, match in enumerate(matches[:3]):  # Show first 3 matches
                    print(f"    Match {i}:")
                    for key, value in match.items():
                        print(f"      {key}: {value}")
            if 'max_confidence' in video_data:
                print(f"  Max confidence: {video_data['max_confidence']}")
    
    return results

if __name__ == "__main__":
    debug_app_processing()