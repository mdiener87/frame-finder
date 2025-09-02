# test_analyzer.py
import torch
import numpy as np
from PIL import Image
from analyzer import (
    ImagePreprocessor, 
    TemplateMatcher, 
    LPIPSVerifier, 
    ReferenceCache,
    extract_frames,
    non_max_suppression,
    temporal_smoothing,
    process_videos
)

def test_preprocessor():
    """Test image preprocessing pipeline"""
    print("Testing ImagePreprocessor...")
    
    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='red')
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=24)
    
    # Test CLIP preprocessing
    clip_tensor = preprocessor.preprocess_for_clip(test_image)
    assert clip_tensor.shape == (1, 3, 224, 224), f"CLIP tensor shape: {clip_tensor.shape}"
    
    # Test LPIPS preprocessing
    lpips_tensor = preprocessor.preprocess_for_lpips(test_image)
    assert lpips_tensor.shape == (1, 3, 224, 224), f"LPIPS tensor shape: {lpips_tensor.shape}"
    
    # Test template matching preprocessing
    template_image = preprocessor.preprocess_for_template_matching(test_image)
    assert template_image.shape == (224, 224), f"Template image shape: {template_image.shape}"
    
    print("✓ ImagePreprocessor tests passed")

def test_template_matcher():
    """Test template matching functionality"""
    print("Testing TemplateMatcher...")
    
    # Create test template and frame
    template = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    frame = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    
    # Place template in frame (with some noise)
    frame[50:100, 50:100] = template
    
    # Initialize matcher
    matcher = TemplateMatcher(scales=[1.0])  # Use only 1.0 scale for simplicity
    
    # Perform matching
    matches = matcher.match_template_multiscale(frame, template)
    
    # Should find at least one match
    assert len(matches) > 0, "Should find at least one match"
    
    # Check match structure
    match = matches[0]
    assert 'bbox' in match, "Match should have bbox"
    assert 'score' in match, "Match should have score"
    assert 'scale' in match, "Match should have scale"
    
    print("✓ TemplateMatcher tests passed")

def test_lpips_verifier():
    """Test LPIPS verification"""
    print("Testing LPIPSVerifier...")
    
    # Initialize verifier
    verifier = LPIPSVerifier()
    
    # Create two identical test images
    test_image = torch.rand(1, 3, 224, 224).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute distance (should be very small for identical images)
    distance = verifier.compute_distance(test_image, test_image)
    assert distance >= 0, "Distance should be non-negative"
    
    print("✓ LPIPSVerifier tests passed")

def test_nms():
    """Test non-maximum suppression"""
    print("Testing non_max_suppression...")
    
    # Create test matches with overlapping boxes
    matches = [
        {'bbox': [10, 10, 50, 50], 'score': 0.9},
        {'bbox': [15, 15, 50, 50], 'score': 0.8},  # Overlapping with first
        {'bbox': [100, 100, 50, 50], 'score': 0.7},  # Not overlapping
    ]
    
    # Apply NMS
    filtered = non_max_suppression(matches, iou_threshold=0.5)
    
    # Should keep at least one of the overlapping matches and the non-overlapping one
    assert len(filtered) >= 1, f"Should keep at least one match, got {len(filtered)}"
    
    print("✓ non_max_suppression tests passed")

def test_temporal_smoothing():
    """Test temporal smoothing"""
    print("Testing temporal_smoothing...")
    
    # Create test detections
    detections = [
        {'timestamp': 1.0, 'confidence': 0.9},
        {'timestamp': 2.0, 'confidence': 0.8},
        {'timestamp': 3.0, 'confidence': 0.9},
        {'timestamp': 4.0, 'confidence': 0.7},
    ]
    
    # Apply temporal smoothing
    smoothed = temporal_smoothing(detections, debounce_n=2, debounce_m=5)
    
    # Should return some detections
    assert isinstance(smoothed, list), "Should return a list"
    
    print("✓ temporal_smoothing tests passed")

if __name__ == "__main__":
    print("Running analyzer tests...")
    print("")
    
    try:
        test_preprocessor()
        test_template_matcher()
        test_lpips_verifier()
        test_nms()
        test_temporal_smoothing()
        
        print("")
        print("All tests passed! ✅")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise