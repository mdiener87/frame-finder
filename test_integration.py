# test_integration.py
import os
import tempfile
import numpy as np
from PIL import Image

def create_test_data():
    """Create simple test images and a test video"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create a simple red square reference image
    ref_image = Image.new('RGB', (100, 100), color='red')
    ref_path = os.path.join(temp_dir, 'reference.png')
    ref_image.save(ref_path)
    
    # Create a test frame with the red square in it
    frame = Image.new('RGB', (400, 300), color='white')
    # Paste the red square in the frame
    frame.paste(ref_image, (150, 100))
    
    # Save as a test "video frame"
    frame_path = os.path.join(temp_dir, 'frame.png')
    frame.save(frame_path)
    
    return [ref_path], [frame_path]

if __name__ == "__main__":
    # This is a simple integration test
    print("Creating test data...")
    ref_paths, frame_paths = create_test_data()
    
    print(f"Reference images: {ref_paths}")
    print(f"Test frames: {frame_paths}")
    print("Test data created successfully!")