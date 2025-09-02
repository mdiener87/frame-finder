# test_regular_analyzer.py
import os
import tempfile
import numpy as np
from PIL import Image
import cv2
from analyzer import process_videos

def create_test_video_and_reference():
    """Create a simple test video and reference image"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create a simple red square reference image (50x50)
    ref_image = Image.new('RGB', (50, 50), color='red')
    ref_path = os.path.join(temp_dir, 'reference.png')
    ref_image.save(ref_path)
    
    # Create a test video with the red square appearing in some frames
    video_path = os.path.join(temp_dir, 'test_video.mp4')
    
    # Video parameters
    width, height = 320, 240
    fps = 10
    duration = 5  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Generate frames
    for i in range(total_frames):
        # Create a white background frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # In some frames, add the red square
        if i in [10, 15, 20, 25, 30]:  # Add square in these frames
            # Place red square at different positions
            x_pos = 100 + (i % 5) * 20
            y_pos = 80 + (i % 3) * 30
            frame[y_pos:y_pos+50, x_pos:x_pos+50] = [0, 0, 255]  # BGR for red
        
        out.write(frame)
    
    out.release()
    
    return [ref_path], [video_path]

def test_regular_pipeline():
    """Test the regular processing pipeline"""
    print("Creating test data...")
    ref_paths, video_paths = create_test_video_and_reference()
    
    print(f"Reference images: {ref_paths}")
    print(f"Test videos: {video_paths}")
    
    print("Running regular processing pipeline...")
    results = process_videos(
        reference_paths=ref_paths,
        video_paths=video_paths,
        frame_interval=0.5,  # Process every 0.5 seconds
        lpips_threshold=0.6,  # Reasonable threshold
        clip_threshold=0.2   # Reasonable threshold
    )
    
    print("Processing complete!")
    print(f"Results: {results}")
    
    # Check results
    video_name = os.path.basename(video_paths[0])
    if video_name in results:
        matches = results[video_name]['matches']
        print(f"Found {len(matches)} matches")
        for match in matches[:5]:  # Show first 5 matches
            if 'error' in match:
                print(f"  Error: {match['error']}")
            else:
                timestamp = match.get('timestamp', 'N/A')
                confidence = match.get('confidence', 'N/A')
                lpips_score = match.get('lpips_score', 'N/A')
                clip_score = match.get('clip_score', 'N/A')
                
                # Format numeric values
                timestamp_str = f"{timestamp:.3f}" if isinstance(timestamp, (int, float)) else str(timestamp)
                confidence_str = f"{confidence:.3f}" if isinstance(confidence, (int, float)) else str(confidence)
                lpips_str = f"{lpips_score:.3f}" if isinstance(lpips_score, (int, float)) else str(lpips_score)
                clip_str = f"{clip_score:.3f}" if isinstance(clip_score, (int, float)) else str(clip_score)
                
                print(f"  Timestamp: {timestamp_str}, "
                      f"Confidence: {confidence_str}, "
                      f"LPIPS: {lpips_str}, "
                      f"CLIP: {clip_str}")
    
    return results

if __name__ == "__main__":
    try:
        results = test_regular_pipeline()
        print("\n✓ Regular pipeline test completed!")
    except Exception as e:
        print(f"\n✗ Regular pipeline test failed: {e}")
        import traceback
        traceback.print_exc()