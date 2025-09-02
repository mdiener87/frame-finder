#!/usr/bin/env python3
"""
Statistical analysis tool to compare positive and negative results.
"""

import numpy as np
from scipy import stats
import os
from frame_analyzer import FrameAnalyzer

def analyze_result_differences(positive_result, negative_result):
    """
    Analyze the statistical significance of differences between positive and negative results.
    """
    print("Statistical Analysis of Results")
    print("=" * 40)
    
    # Extract key metrics
    pos_max = positive_result['max_similarity']
    pos_mean = positive_result['mean_similarity']
    pos_std = positive_result['std_similarity']
    
    neg_max = negative_result['max_similarity']
    neg_mean = negative_result['mean_similarity']
    neg_std = negative_result['std_similarity']
    
    print(f"Positive Result:")
    print(f"  Max Similarity: {pos_max:.4f}")
    print(f"  Mean Similarity: {pos_mean:.4f}")
    print(f"  Std Deviation: {pos_std:.4f}")
    
    print(f"\nNegative Result:")
    print(f"  Max Similarity: {neg_max:.4f}")
    print(f"  Mean Similarity: {neg_mean:.4f}")
    print(f"  Std Deviation: {neg_std:.4f}")
    
    # Calculate differences
    max_diff = pos_max - neg_max
    mean_diff = pos_mean - neg_mean
    
    print(f"\nDifferences:")
    print(f"  Max Similarity Difference: {max_diff:.4f}")
    print(f"  Mean Similarity Difference: {mean_diff:.4f}")
    
    # Perform statistical tests if we have the raw data
    # (This would require storing all similarity scores, which we don't currently do)
    
    # Simple heuristic: if the difference is larger than the average std deviation, 
    # it might be significant
    avg_std = (pos_std + neg_std) / 2
    if max_diff > avg_std:
        print(f"\nResult: Differences appear significant (diff={max_diff:.4f} > avg_std={avg_std:.4f})")
    else:
        print(f"\nResult: Differences may not be significant (diff={max_diff:.4f} <= avg_std={avg_std:.4f})")
    
    # Threshold-based classification
    classification_threshold = 0.35  # Adjustable threshold
    
    pos_found = pos_max > classification_threshold
    neg_found = neg_max > classification_threshold
    
    print(f"\nThreshold-based Classification (threshold={classification_threshold}):")
    print(f"  Positive classified as FOUND: {pos_found}")
    print(f"  Negative classified as FOUND: {neg_found}")
    
    if pos_found and not neg_found:
        print("  Result: Clear distinction between positive and negative")
    elif pos_found and neg_found:
        print("  Result: Both classified as positive - potential false positive in negative")
    elif not pos_found and not neg_found:
        print("  Result: Both classified as negative - potential false negative in positive")
    else:
        print("  Result: Negative classified as positive, positive as negative - concerning result")

def main():
    """Main function to demonstrate statistical analysis."""
    print("Frame Finder - Statistical Analysis Tool")
    print("=" * 50)
    
    # This would normally be done with actual results from your analysis
    # For demonstration, let's create mock results based on your examples:
    
    # Example from your observation:
    # Positive (Flesh and Blood): max confidence 83.97%
    # Negative (Renaissance Man): max confidence 79.59%
    
    positive_result = {
        'max_similarity': 0.8397,
        'mean_similarity': 0.65,
        'std_similarity': 0.08
    }
    
    negative_result = {
        'max_similarity': 0.7959,
        'mean_similarity': 0.62,
        'std_similarity': 0.07
    }
    
    analyze_result_differences(positive_result, negative_result)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()