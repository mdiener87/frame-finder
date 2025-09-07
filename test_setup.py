#!/usr/bin/env python3
"""
Test script to verify that all dependencies are properly installed
and the basic functionality works.
"""

import os
import sys

def test_dependencies():
    """Test if all required dependencies are available"""
    try:
        import flask
        print("✓ Flask installed")
    except ImportError:
        print("✗ Flask not installed")
        return False
    
    try:
        import cv2
        print("✓ OpenCV installed")
    except ImportError:
        print("✗ OpenCV not installed")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL/Pillow installed")
    except ImportError:
        print("✗ PIL/Pillow not installed")
        return False
    
    try:
        import numpy
        print("✓ NumPy installed")
    except ImportError:
        print("✗ NumPy not installed")
        return False
    
    # PyTorch/Transformers no longer required for the analyzer
    
    return True

def test_cv_features():
    """Basic OpenCV feature detection sanity check."""
    try:
        import numpy as np
        import cv2
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
        orb = cv2.ORB_create(nfeatures=500)
        kp, desc = orb.detectAndCompute(img, None)
        assert kp is not None and desc is not None
        print("✓ OpenCV ORB features computed")
        return True
    except Exception as e:
        print(f"✗ OpenCV feature test failed: {e}")
        return False

def test_file_structure():
    """Test if required files and directories exist"""
    required_files = [
        'app.py',
        'analyzer.py',
        'requirements.txt'
    ]
    
    required_dirs = [
        'templates',
        'static',
        'static/css',
        'static/js',
        'static/thumbnails'
    ]
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_good = False
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory} exists")
        else:
            print(f"✗ {directory} missing")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("Testing Frame Finder setup...\n")
    
    print("1. Testing dependencies:")
    deps_ok = test_dependencies()
    
    print("\n2. Testing OpenCV features:")
    model_ok = test_cv_features()
    
    print("\n3. Testing file structure:")
    structure_ok = test_file_structure()
    
    print("\n" + "="*50)
    if deps_ok and model_ok and structure_ok:
        print("✓ All tests passed! Frame Finder is ready to run.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
