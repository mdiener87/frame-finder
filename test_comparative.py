#!/usr/bin/env python3
"""
Test script to validate the batch analysis approach with your specific example.
"""

from frame_analyzer import FrameAnalyzer
import os

def test_comparative_analysis():
    """Test comparative analysis with the example videos."""
    # Initialize analyzer
    analyzer = FrameAnalyzer()
    
    # Define paths (you would update these to your actual file paths)
    reference_path = "examples/thinktank/ReferenceImage.png"
    
    # For demonstration, let's create mock results based on your actual results
    print("Testing Comparative Analysis Approach")
    print("=" * 50)
    
    # Mock results based on your observations:
    # Flesh and Blood: Max confidence 83.97%, 3 matches
    # Renaissance Man: Max confidence 79.59%, 0 matches
    
    flesh_and_blood_result = {
        'max_similarity': 0.8397,
        'mean_similarity': 0.65,
        'std_similarity': 0.08,
        'matches': [{'timestamp': 1.0, 'similarity': 0.8}] * 3,  # 3 matches
        'confidence': 0.8397,
        'found': True
    }
    
    renaissance_man_result = {
        'max_similarity': 0.7959,
        'mean_similarity': 0.62,
        'std_similarity': 0.07,
        'matches': [],  # 0 matches
        'confidence': 0.7959,
        'found': False
    }
    
    # Perform comparative analysis
    results = {
        'Star_Trek_Voyager_S07e09-10_Flesh_And_Blood.mp4': flesh_and_blood_result,
        'Star_Trek_Voyager_S07e24_Renaissance_Man.mp4': renaissance_man_result
    }
    
    # Extract metrics
    video_metrics = {}
    for video_name, result in results.items():
        video_metrics[video_name] = {
            'max_similarity': result['max_similarity'],
            'matches_count': len(result['matches'])
        }
    
    print("Video Analysis Results:")
    print("-" * 30)
    for video_name, metrics in video_metrics.items():
        print(f"{video_name}:")
        print(f"  Max Similarity: {metrics['max_similarity']:.4f}")
        print(f"  Matches Count: {metrics['matches_count']}")
        print()
    
    # Comparative analysis
    print("Comparative Analysis:")
    print("-" * 30)
    
    # Calculate differences
    max_similarity = max([m['max_similarity'] for m in video_metrics.values()])
    max_matches = max([m['matches_count'] for m in video_metrics.values()])
    
    for video_name, metrics in video_metrics.items():
        matches_ratio = metrics['matches_count'] / max(1, max_matches) if max_matches > 0 else 0
        similarity_ratio = metrics['max_similarity'] / max(1, max_similarity) if max_similarity > 0 else 0
        
        # Criteria for being "likely"
        has_significant_matches = matches_ratio > 0.5
        has_high_similarity = similarity_ratio > 0.9
        absolute_high_confidence = metrics['max_similarity'] > 0.80
        
        is_likely = (
            (has_significant_matches and has_high_similarity) or
            absolute_high_confidence or
            (metrics['matches_count'] > 0 and metrics['max_similarity'] > 0.75)
        )
        
        print(f"{video_name}:")
        print(f"  Matches Ratio: {matches_ratio:.2f}")
        print(f"  Similarity Ratio: {similarity_ratio:.2f}")
        print(f"  Has Significant Matches: {has_significant_matches}")
        print(f"  Has High Similarity: {has_high_similarity}")
        print(f"  Absolute High Confidence: {absolute_high_confidence}")
        print(f"  --> Is Likely Match: {is_likely}")
        print()
    
    print("CONCLUSION:")
    print("-" * 30)
    likely_matches = [name for name, metrics in video_metrics.items() 
                     if metrics['matches_count'] / max(1, max_matches) > 0.5 or metrics['max_similarity'] > 0.80]
    
    if likely_matches:
        print(f"Most likely to contain reference: {likely_matches[0]}")
        print("Reasoning: Higher match count (3 vs 0) and high similarity score (83.97% vs 79.59%)")
    else:
        print("Cannot reliably distinguish between videos")

if __name__ == "__main__":
    test_comparative_analysis()