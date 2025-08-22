#!/usr/bin/env python3
"""
Script to run the Frame Finder application.
This script provides instructions on how to set up and run the application.
"""

import sys
import os

def main():
    print("Frame Finder - Visual Prop Identification Tool")
    print("=" * 50)
    print()
    
    print("To run Frame Finder, follow these steps:")
    print()
    
    print("1. Create a virtual environment:")
    print("   python3 -m venv venv")
    print()
    
    print("2. Activate the virtual environment:")
    print("   On Linux/Mac: source venv/bin/activate")
    print("   On Windows: venv\\Scripts\\activate")
    print()
    
    print("3. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    
    print("4. Run the application:")
    print("   python app.py")
    print()
    
    print("5. Open your browser to http://localhost:5000")
    print()
    
    print("Note: The first time you run the application,")
    print("it will download the CLIP model which may take a few minutes.")
    print()
    
    print("Press Ctrl+C to exit this script.")

if __name__ == "__main__":
    main()