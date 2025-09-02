#!/usr/bin/env python3
"""
Video Comparison Tool - Compares two videos to determine which is more likely to contain the reference.
"""

import numpy as np
from frame_analyzer import FrameAnalyzer
import os

class VideoComparator:
    def __init__(self):
        """Initialize the comparator with frame analyzer."""
        self.analyzer = FrameAnalyzer()
    
    def analyze_video(self, reference_path: str, video_path: str, frame_interval: float = 1.0):
        """
        Analyze a single video and return detailed statistics.
        """
        print(f"Analyzing {os.path.basename(video_path)}...")
        result = self.analyzer.detect_reference_in_video(
            reference_path, 
            video_path, 
            frame_interval=frame_interval
        )
        return result
    
    def compare_videos(self, reference_path: str, video1_path: str, video2_path: str, 
                      frame_interval: float = 1.0):
        """
        Compare two videos to determine which is more likely to contain the reference.
        """
        print("="*70)
        print("VIDEO COMPARATOR - Determining which video contains the reference")
        print("="*70)
        print(f"Reference: {os.path.basename(reference_path)}")
        print(f"Video 1: {os.path.basename(video1_path)}")
        print(f"Video 2: {os.path.basename(video2_path)}")
        print("-"*70)
        
        # Analyze both videos
        result1 = self.analyze_video(reference_path, video1_path, frame_interval)
        print()
        result2 = self.analyze_video(reference_path, video2_path, frame_interval)
        print()
        
        # Extract key metrics
        metrics1 = {
            'max_similarity': result1['max_similarity'],
            'mean_similarity': result1['mean_similarity'],
            'std_similarity': result1['std_similarity'],
            'matches_count': len(result1['matches']),
            'confidence': result1['confidence'],
            'found': result1['found']
        }
        
        metrics2 = {
            'max_similarity': result2['max_similarity'],
            'mean_similarity': result2['mean_similarity'],
            'std_similarity': result2['std_similarity'],
            'matches_count': len(result2['matches']),
            'confidence': result2['confidence'],
            'found': result2['found']
        }
        
        # Display comparison
        print("COMPARISON RESULTS")
        print("-"*70)
        print(f"{'Metric':<20} {'Video 1':<15} {'Video 2':<15} {'Difference':<15}")
        print("-"*70)
        
        max_diff = abs(metrics1['max_similarity'] - metrics2['max_similarity'])
        mean_diff = abs(metrics1['mean_similarity'] - metrics2['mean_similarity'])
        matches_diff = abs(metrics1['matches_count'] - metrics2['matches_count'])
        
        print(f"{'Max Similarity':<20} {metrics1['max_similarity']:<15.4f} {metrics2['max_similarity']:<15.4f} {max_diff:<15.4f}")
        print(f"{'Mean Similarity':<20} {metrics1['mean_similarity']:<15.4f} {metrics2['mean_similarity']:<15.4f} {mean_diff:<15.4f}")
        print(f"{'Std Deviation':<20} {metrics1['std_similarity']:<15.4f} {metrics2['std_similarity']:<15.4f} {abs(metrics1['std_similarity'] - metrics2['std_similarity']):<15.4f}")
        print(f"{'Matches Count':<20} {metrics1['matches_count']:<15d} {metrics2['matches_count']:<15d} {matches_diff:<15d}")
        print(f"{'Confidence':<20} {metrics1['confidence']:<15.4f} {metrics2['confidence']:<15.4f} {abs(metrics1['confidence'] - metrics2['confidence']):<15.4f}")
        print(f"{'Found (Bool)':<20} {str(metrics1['found']):<15} {str(metrics2['found']):<15} {'N/A':<15}")
        
        # Statistical significance test
        print("\nSTATISTICAL ANALYSIS")
        print("-"*70)
        
        # If one has significantly more matches, that's a strong indicator
        if matches_diff > min(metrics1['matches_count'], metrics2['matches_count']) * 0.5:
            if metrics1['matches_count'] > metrics2['matches_count']:
                likely_video = "Video 1"
                ratio = metrics1['matches_count'] / max(1, metrics2['matches_count'])
            else:
                likely_video = "Video 2"
                ratio = metrics2['matches_count'] / max(1, metrics1['matches_count'])
            print(f"✓ Matches count difference significant (ratio: {ratio:.2f}:1)")
        else:
            print("✗ Matches count difference not significant")
        
        # Max similarity difference analysis
        if max_diff > 0.05:  # 5% difference threshold
            if metrics1['max_similarity'] > metrics2['max_similarity']:
                max_likelihood = "Video 1"
            else:
                max_likelihood = "Video 2"
            print(f"✓ Max similarity difference significant ({max_diff:.4f}) - favors {max_likelihood}")
        else:
            print(f"✗ Max similarity difference not significant ({max_diff:.4f})")
        
        # Mean similarity difference analysis
        if mean_diff > 0.03:  # 3% difference threshold
            if metrics1['mean_similarity'] > metrics2['mean_similarity']:
                mean_likelihood = "Video 1"
            else:
                mean_likelihood = "Video 2"
            print(f"✓ Mean similarity difference significant ({mean_diff:.4f}) - favors {mean_likelihood}")
        else:
            print(f"✗ Mean similarity difference not significant ({mean_diff:.4f})")
        
        # Confidence-based decision
        print("\nCONFIDENCE-BASED DECISION")
        print("-"*70)
        
        # Use a higher threshold for more reliable results
        threshold = 0.80
        
        v1_high_conf = metrics1['max_similarity'] > threshold
        v2_high_conf = metrics2['max_similarity'] > threshold
        
        if v1_high_conf and not v2_high_conf:
            decision = "Video 1 is more likely to contain the reference"
        elif v2_high_conf and not v1_high_conf:
            decision = "Video 2 is more likely to contain the reference"
        elif v1_high_conf and v2_high_conf:
            # Both have high confidence - look at which is higher
            if metrics1['max_similarity'] > metrics2['max_similarity']:
                decision = "Video 1 is more likely (higher max similarity)"
            else:
                decision = "Video 2 is more likely (higher max similarity)"
        else:
            # Neither has high confidence - use relative comparison
            if max_diff > 0.03:  # Minimum meaningful difference
                if metrics1['max_similarity'] > metrics2['max_similarity']:
                    decision = "Video 1 is more likely (relative comparison)"
                else:
                    decision = "Video 2 is more likely (relative comparison)"
            else:
                decision = "Cannot reliably distinguish between videos - both low confidence"
        
        print(f"Decision threshold: {threshold}")
        print(f"Video 1 above threshold: {v1_high_conf}")
        print(f"Video 2 above threshold: {v2_high_conf}")
        print(f"FINAL DECISION: {decision}")
        
        return {
            'video1': metrics1,
            'video2': metrics2,
            'decision': decision
        }

def main():
    """Main function to demonstrate video comparison."""
    comparator = VideoComparator()
    
    # Example usage with your files
    # You would replace these paths with your actual file paths
    reference_path = "examples/thinktank/ReferenceImage.png"
    
    # For demonstration, let's create a mock comparison based on your results
    print("MOCK COMPARISON EXAMPLE")
    print("Based on your actual results:")
    print("- Flesh and Blood: Max confidence 83.97%")
    print("- Renaissance Man: Max confidence 79.59%")
    print()
    
    # Create mock results to demonstrate the comparison logic
    mock_result1 = {
        'max_similarity': 0.8397,
        'mean_similarity': 0.65,
        'std_similarity': 0.08,
        'matches': [{'timestamp': 1.0, 'similarity': 0.8}] * 3,  # 3 matches
        'confidence': 0.8397,
        'found': True
    }
    
    mock_result2 = {
        'max_similarity': 0.7959,
        'mean_similarity': 0.62,
        'std_similarity': 0.07,
        'matches': [],  # 0 matches
        'confidence': 0.7959,
        'found': False
    }
    
    print("="*70)
    print("VIDEO COMPARATOR - Mock Analysis")
    print("="*70)
    print(f"Reference: ReferenceImage.png")
    print(f"Video 1: Star_Trek_Voyager_S07e09-10_Flesh_And_Blood.mp4")
    print(f"Video 2: Star_Trek_Voyager_S07e24_Renaissance_Man.mp4")
    print("-"*70)
    
    # Display mock comparison
    metrics1 = {
        'max_similarity': mock_result1['max_similarity'],
        'mean_similarity': mock_result1['mean_similarity'],
        'std_similarity': mock_result1['std_similarity'],
        'matches_count': len(mock_result1['matches']),
        'confidence': mock_result1['confidence'],
        'found': mock_result1['found']
    }
    
    metrics2 = {
        'max_similarity': mock_result2['max_similarity'],
        'mean_similarity': mock_result2['mean_similarity'],
        'std_similarity': mock_result2['std_similarity'],
        'matches_count': len(mock_result2['matches']),
        'confidence': mock_result2['confidence'],
        'found': mock_result2['found']
    }
    
    print("COMPARISON RESULTS")
    print("-"*70)
    print(f"{'Metric':<20} {'Video 1':<15} {'Video 2':<15} {'Difference':<15}")
    print("-"*70)
    
    max_diff = abs(metrics1['max_similarity'] - metrics2['max_similarity'])
    mean_diff = abs(metrics1['mean_similarity'] - metrics2['mean_similarity'])
    matches_diff = abs(metrics1['matches_count'] - metrics2['matches_count'])
    
    print(f"{'Max Similarity':<20} {metrics1['max_similarity']:<15.4f} {metrics2['max_similarity']:<15.4f} {max_diff:<15.4f}")
    print(f"{'Mean Similarity':<20} {metrics1['mean_similarity']:<15.4f} {metrics2['mean_similarity']:<15.4f} {mean_diff:<15.4f}")
    print(f"{'Std Deviation':<20} {metrics1['std_similarity']:<15.4f} {metrics2['std_similarity']:<15.4f} {abs(metrics1['std_similarity'] - metrics2['std_similarity']):<15.4f}")
    print(f"{'Matches Count':<20} {metrics1['matches_count']:<15d} {metrics2['matches_count']:<15d} {matches_diff:<15d}")
    print(f"{'Confidence':<20} {metrics1['confidence']:<15.4f} {metrics2['confidence']:<15.4f} {abs(metrics1['confidence'] - metrics2['confidence']):<15.4f}")
    print(f"{'Found (Bool)':<20} {str(metrics1['found']):<15} {str(metrics2['found']):<15} {'N/A':<15}")
    
    print("\nSTATISTICAL ANALYSIS")
    print("-"*70)
    
    # Matches count analysis
    if matches_diff > 0:
        if metrics1['matches_count'] > metrics2['matches_count']:
            likely_video = "Video 1"
            ratio = metrics1['matches_count'] / max(1, metrics2['matches_count'])
        else:
            likely_video = "Video 2"
            ratio = metrics2['matches_count'] / max(1, metrics1['matches_count'])
        print(f"✓ Matches count difference significant (ratio: {ratio:.2f}:1) - favors {likely_video}")
    else:
        print("✗ Matches count difference not significant")
    
    # Max similarity analysis
    if max_diff > 0.05:
        if metrics1['max_similarity'] > metrics2['max_similarity']:
            max_likelihood = "Video 1"
        else:
            max_likelihood = "Video 2"
        print(f"✓ Max similarity difference significant ({max_diff:.4f}) - favors {max_likelihood}")
    else:
        print(f"✗ Max similarity difference not significant ({max_diff:.4f})")
    
    # Mean similarity analysis
    if mean_diff > 0.03:
        if metrics1['mean_similarity'] > metrics2['mean_similarity']:
            mean_likelihood = "Video 1"
        else:
            mean_likelihood = "Video 2"
        print(f"✓ Mean similarity difference significant ({mean_diff:.4f}) - favors {mean_likelihood}")
    else:
        print(f"✗ Mean similarity difference not significant ({mean_diff:.4f})")
    
    print("\nCONFIDENCE-BASED DECISION")
    print("-"*70)
    
    threshold = 0.80
    v1_high_conf = metrics1['max_similarity'] > threshold
    v2_high_conf = metrics2['max_similarity'] > threshold
    
    print(f"Decision threshold: {threshold}")
    print(f"Video 1 above threshold: {v1_high_conf}")
    print(f"Video 2 above threshold: {v2_high_conf}")
    
    if v1_high_conf and not v2_high_conf:
        decision = "Video 1 is more likely to contain the reference"
    elif v2_high_conf and not v1_high_conf:
        decision = "Video 2 is more likely to contain the reference"
    elif v1_high_conf and v2_high_conf:
        if metrics1['max_similarity'] > metrics2['max_similarity']:
            decision = "Video 1 is more likely (higher max similarity)"
        else:
            decision = "Video 2 is more likely (higher max similarity)"
    else:
        if max_diff > 0.03:
            if metrics1['max_similarity'] > metrics2['max_similarity']:
                decision = "Video 1 is more likely (relative comparison)"
            else:
                decision = "Video 2 is more likely (relative comparison)"
        else:
            decision = "Cannot reliably distinguish between videos - both low confidence"
    
    print(f"FINAL DECISION: {decision}")

if __name__ == "__main__":
    main()