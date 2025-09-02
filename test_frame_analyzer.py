#!/usr/bin/env python3
"""
Unit tests for the frame analyzer.
"""

import unittest
import os
import sys
import numpy as np
from PIL import Image

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frame_analyzer import FrameAnalyzer

class TestFrameAnalyzer(unittest.TestCase):
    """Test cases for the FrameAnalyzer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.analyzer = FrameAnalyzer()
        cls.test_data_path = "examples/thinktank"
        cls.reference_image_path = os.path.join(cls.test_data_path, "ReferenceImage.png")
        cls.positive_video_path = os.path.join(cls.test_data_path, "TT_Positive.mp4")
        cls.negative_video_path = os.path.join(cls.test_data_path, "TT_Negative.mp4")
    
    def test_reference_image_exists(self):
        """Test that the reference image file exists."""
        self.assertTrue(os.path.exists(self.reference_image_path), 
                       "Reference image file should exist")
    
    def test_video_files_exist(self):
        """Test that both test video files exist."""
        self.assertTrue(os.path.exists(self.positive_video_path), 
                       "Positive test video file should exist")
        self.assertTrue(os.path.exists(self.negative_video_path), 
                       "Negative test video file should exist")
    
    def test_preprocess_image(self):
        """Test that image preprocessing works correctly."""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Preprocess it
        processed = self.analyzer.preprocess_image(test_image)
        
        # Check that it's been resized to the expected size
        self.assertEqual(processed.size, (224, 224), 
                        "Processed image should be 224x224 pixels")
        
        # Check that it's in RGB mode
        self.assertEqual(processed.mode, 'RGB', 
                        "Processed image should be in RGB mode")
    
    def test_extract_frames(self):
        """Test frame extraction from video."""
        if not os.path.exists(self.positive_video_path):
            self.skipTest("Positive video file not found")
            
        # Extract frames with a 1-second interval
        frames = self.analyzer.extract_frames(self.positive_video_path, frame_interval=1.0)
        
        # Should have at least some frames
        self.assertGreater(len(frames), 0, "Should extract at least one frame")
        
        # Check that each item is a tuple of (Image, timestamp)
        for frame, timestamp in frames:
            self.assertIsInstance(frame, Image.Image, 
                                "First element should be a PIL Image")
            self.assertIsInstance(timestamp, float, 
                                "Second element should be a timestamp float")
            self.assertGreaterEqual(timestamp, 0, 
                                  "Timestamp should be non-negative")
    
    def test_compute_similarity(self):
        """Test similarity computation between images."""
        if not os.path.exists(self.reference_image_path):
            self.skipTest("Reference image file not found")
            
        # Load the reference image
        ref_image = Image.open(self.reference_image_path)
        
        # Compare with itself - should have high similarity
        similarity = self.analyzer.compute_similarity(ref_image, ref_image)
        
        # Self-similarity should be very high (close to 1.0)
        self.assertGreater(similarity, 0.9, 
                          "Self-similarity should be high (> 0.9)")
        self.assertLessEqual(similarity, 1.0, 
                           "Similarity should not exceed 1.0")
    
    def test_detect_reference_in_positive_video(self):
        """Test detection of reference image in positive video."""
        if not os.path.exists(self.reference_image_path) or not os.path.exists(self.positive_video_path):
            self.skipTest("Required test files not found")
            
        # Run detection
        result = self.analyzer.detect_reference_in_video(
            self.reference_image_path,
            self.positive_video_path,
            frame_interval=1.0  # Faster test with fewer frames
        )
        
        # Should find the reference image
        self.assertTrue(result['found'], 
                       "Reference image should be found in positive video")
        
        # Should have confidence > 0
        self.assertGreater(result['confidence'], 0, 
                          "Confidence should be greater than 0")
        
        # Should have at least one match
        self.assertGreater(len(result['matches']), 0, 
                          "Should have at least one match")
        
        # Max similarity should be reasonable
        self.assertGreater(result['max_similarity'], 0.4, 
                          "Max similarity should be reasonable")
        
        # Check that stats are included
        self.assertIn('stats', result, "Result should include stats")
        self.assertIn('significant_peak', result['stats'], "Stats should include significant_peak")
    
    def test_detect_reference_in_negative_video(self):
        """Test that reference image is NOT found in negative video."""
        if not os.path.exists(self.reference_image_path) or not os.path.exists(self.negative_video_path):
            self.skipTest("Required test files not found")
            
        # Run detection
        result = self.analyzer.detect_reference_in_video(
            self.reference_image_path,
            self.negative_video_path,
            frame_interval=1.0  # Faster test with fewer frames
        )
        
        # Should NOT find the reference image
        self.assertFalse(result['found'], 
                        "Reference image should NOT be found in negative video")
        
        # Should have low confidence
        self.assertEqual(result['confidence'], 0, 
                       "Confidence should be zero in negative video")
        
        # Should have no matches
        self.assertEqual(len(result['matches']), 0, 
                        "Should have no matches in negative video")

def run_tests():
    """Run all tests and return results."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFrameAnalyzer)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("Running Frame Analyzer Tests...")
    print("=" * 50)
    
    success = run_tests()
    
    print("\n" + "=" * 50)
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)