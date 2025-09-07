#!/usr/bin/env python3
"""
Test script to verify different analysis modes work correctly.
"""

import os
import sys
from frame_analyzer import FrameAnalyzer

def test_modes():
    """Test different analysis modes with example files."""
    # Initialize analyzer
    analyzer = FrameAnalyzer()
    
    # Define paths
    base_path = "examples/thinktank"
    reference_path = os.path.join(base_path, "ReferenceImage.png")
    positive_video = os.path.join(base_path, "TT_Positive.mp4")
    negative_video = os.path.join(base_path, "TT_Negative.mp4")
    
    # Check if files exist
    if not os.path.exists(reference_path):
        print(f"Reference image not found: {reference_path}")
        return
        
    if not os.path.exists(positive_video):
        print(f"Positive video not found: {positive_video}")
        return
        
    if not os.path.exists(negative_video):
        print(f"Negative video not found: {negative_video}")
        return
    
    print("="*60)
    print("TESTING DIFFERENT ANALYSIS MODES")
    print("="*60)
    
    # Test modes with different frame intervals
    modes = [
        ("Standard", 0.5),      # Standard mode
        ("Enhanced", 0.25),    # Enhanced mode
        ("High Precision", 0.1) # High precision mode
    ]
    
    for mode_name, frame_interval in modes:
        print(f"\nTesting {mode_name} Mode (Frame Interval: {frame_interval}s)")
        print("-" * 50)
        
        # Test positive video
        print(f"Positive Video ({os.path.basename(positive_video)}):")
        try:
            result_pos = analyzer.detect_reference_in_video(
                reference_path, 
                positive_video, 
                frame_interval=frame_interval
            )
            
            print(f"  Reference found: {result_pos['found']}")
            print(f"  Confidence: {result_pos['confidence']:.3f}")
            print(f"  Max similarity: {result_pos['max_similarity']:.3f}")
            print(f"  Matches found: {len(result_pos['matches'])}")
            print(f"  Total frames processed: {result_pos['total_frames_processed']}")
            
            if result_pos['matches']:
                # Show top 5 timestamps
                top_timestamps = [f"{m['timestamp']:.2f}" for m in result_pos['matches'][:5]]
                print(f"  Top timestamps (s): {top_timestamps}")
                
        except Exception as e:
            print(f"  Error: {str(e)}")
        
        # Test negative video
        print(f"Negative Video ({os.path.basename(negative_video)}):")
        try:
            result_neg = analyzer.detect_reference_in_video(
                reference_path, 
                negative_video, 
                frame_interval=frame_interval
            )
            
            print(f"  Reference found: {result_neg['found']}")
            print(f"  Confidence: {result_neg['confidence']:.3f}")
            print(f"  Max similarity: {result_neg['max_similarity']:.3f}")
            print(f"  Matches found: {len(result_neg['matches'])}")
            print(f"  Total frames processed: {result_neg['total_frames_processed']}")
            
            if result_neg['matches']:
                # Show top 5 timestamps
                top_timestamps = [f"{m['timestamp']:.2f}" for m in result_neg['matches'][:5]]
                print(f"  Top timestamps (s): {top_timestamps}")
                
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_modes()