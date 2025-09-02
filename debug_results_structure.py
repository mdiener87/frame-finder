# debug_results_structure.py
import json
from analyzer import process_videos
import tempfile
import os
import numpy as np
from PIL import Image

def create_simple_test():
    """Create a simple test to debug results structure"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create a simple red square reference image (50x50)
    ref_image = Image.new('RGB', (50, 50), color='red')
    ref_path = os.path.join(temp_dir, 'reference.png')
    ref_image.save(ref_path)
    
    # Create a simple test "video" as a single frame
    frame = Image.new('RGB', (320, 240), color='white')
    # Paste the red square in the frame
    frame.paste(ref_image, (100, 80))
    
    frame_path = os.path.join(temp_dir, 'test_frame.png')
    frame.save(frame_path)
    
    # For testing, we'll treat this as a video by creating a results dict
    # that mimics what the video processing would return
    return [ref_path], [frame_path]

def debug_results():
    """Debug the results structure"""
    print("Creating test data...")
    ref_paths, frame_paths = create_simple_test()
    
    print(f"Reference paths: {ref_paths}")
    print(f"Frame paths: {frame_paths}")
    
    # Process as if they were videos
    print("Processing with analyzer...")
    results = process_videos(
        reference_paths=ref_paths,
        video_paths=frame_paths,
        frame_interval=1.0,
        lpips_threshold=0.7,  # More lenient for testing
        clip_threshold=0.1   # More lenient for testing
    )
    
    print("Results structure:")
    print(json.dumps(results, indent=2, default=str))
    
    # Check if we have the expected structure
    for video_name, video_data in results.items():
        print(f"\nVideo: {video_name}")
        print(f"  Type of video_data: {type(video_data)}")
        if isinstance(video_data, dict):
            print(f"  Keys: {list(video_data.keys())}")
            if 'matches' in video_data:
                matches = video_data['matches']
                print(f"  Number of matches: {len(matches)}")
                for i, match in enumerate(matches[:3]):  # Show first 3 matches
                    print(f"    Match {i}: {match}")
            if 'max_confidence' in video_data:
                print(f"  Max confidence: {video_data['max_confidence']}")
    
    return results

if __name__ == "__main__":
    debug_results()