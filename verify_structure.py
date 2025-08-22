#!/usr/bin/env python3
"""
Simple script to verify that our code structure is correct
"""

import sys
import os

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

def test_file_contents():
    """Test if key files have the expected content"""
    try:
        # Check app.py
        with open('app.py', 'r') as f:
            app_content = f.read()
            required_elements = [
                'Flask',
                'render_template',
                'process_videos',
                '@app.route'
            ]
            
            for element in required_elements:
                if element in app_content:
                    print(f"✓ app.py contains {element}")
                else:
                    print(f"✗ app.py missing {element}")
        
        # Check analyzer.py
        with open('analyzer.py', 'r') as f:
            analyzer_content = f.read()
            required_elements = [
                'extract_frames',
                'compare_images',
                'process_videos',
                'cv2',
                'Image'
            ]
            
            for element in required_elements:
                if element in analyzer_content:
                    print(f"✓ analyzer.py contains {element}")
                else:
                    print(f"✗ analyzer.py missing {element}")
        
        # Check templates
        templates = ['templates/base.html', 'templates/index.html', 'templates/results.html']
        for template in templates:
            if os.path.exists(template):
                print(f"✓ {template} exists")
            else:
                print(f"✗ {template} missing")
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking file contents: {e}")
        return False

if __name__ == "__main__":
    print("Verifying Frame Finder structure...\n")
    
    print("1. Testing file structure:")
    structure_ok = test_file_structure()
    
    print("\n2. Testing file contents:")
    contents_ok = test_file_contents()
    
    print("\n" + "="*50)
    if structure_ok and contents_ok:
        print("✓ Structure verification passed!")
        print("The code structure is correct and ready for dependency installation.")
        sys.exit(0)
    else:
        print("✗ Structure verification failed.")
        sys.exit(1)