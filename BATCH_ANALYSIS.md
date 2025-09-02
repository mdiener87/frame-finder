# Frame Finder - Batch Analysis Approach

## Overview

This document explains the new batch analysis approach for determining which videos contain a reference image, without requiring manual confidence threshold adjustments.

## The Problem

When analyzing videos for reference images, we often encounter cases where:
- Two videos have very similar confidence scores (e.g., 83.97% vs 79.59%)
- It's difficult to distinguish between positive and negative results without prior knowledge
- Manual threshold adjustment requires knowing what the "floor" should be

## The Solution

Our new approach uses **comparative analysis** to automatically determine which videos are most likely to contain the reference image.

### Key Principles

1. **Multiple Metrics**: We don't rely on a single similarity score, but consider:
   - Match count (number of frames that exceed threshold)
   - Maximum similarity score
   - Mean similarity score
   - Statistical distribution of scores

2. **Relative Comparison**: When analyzing multiple videos, we compare them relative to each other rather than against an absolute standard.

3. **Automatic Thresholding**: The system determines likely matches without requiring manual threshold adjustment.

## How It Works

### Example Case: Flesh and Blood vs Renaissance Man

```
Video 1: Star_Trek_Voyager_S07e09-10_Flesh_And_Blood.mp4
- Max Similarity: 83.97%
- Matches Found: 3
- Confidence Level: High

Video 2: Star_Trek_Voyager_S07e24_Renaissance_Man.mp4  
- Max Similarity: 79.59%
- Matches Found: 0
- Confidence Level: Low
```

### Comparative Analysis Process

1. **Match Count Analysis**: 
   - Flesh and Blood: 3 matches
   - Renaissance Man: 0 matches
   - Ratio: 3:0 = Infinite difference (very significant)

2. **Similarity Score Analysis**:
   - Difference: 83.97% - 79.59% = 4.38%
   - While small in absolute terms, this is significant when combined with other metrics

3. **Confidence Threshold**:
   - Videos exceeding 80% similarity are considered high confidence
   - Flesh and Blood: 83.97% > 80% ✓
   - Renaissance Man: 79.59% < 80% ✗

4. **Final Determination**:
   - Flesh and Blood has significantly more matches AND exceeds confidence threshold
   - Renaissance Man has no matches AND falls below confidence threshold
   - **Conclusion**: Flesh and Blood is the most likely to contain the reference

## Implementation Details

### Backend Logic

The system uses the following criteria to determine likely matches:

1. **Primary Criteria** (ANY must be true):
   - Has significant matches (more than 50% of the best match count) AND high similarity (within 10% of best similarity)
   - Absolute high confidence (max similarity > 80%)
   - Some matches with decent similarity (matches > 0 AND max similarity > 75%)

2. **Confidence Levels**:
   - High: Max similarity > 85%
   - Medium: Max similarity > 75%
   - Low: Max similarity ≤ 75%

### Frontend Interface

The new batch analysis interface:

1. **Upload Reference Image**: Drag and drop or select the reference image
2. **Select Multiple Videos**: Upload several video files for analysis  
3. **Automatic Analysis**: System processes all videos and compares results
4. **Clear Results**: Simple YES/NO determination for each video
5. **Detailed Explanation**: Shows reasoning behind each determination

## Benefits

1. **No Manual Threshold Adjustment**: System automatically determines likely matches
2. **Comparative Intelligence**: Uses relative differences between videos
3. **Multiple Validation**: Considers several metrics together
4. **Clear Results**: Simple yes/no answers without ambiguity
5. **Detailed Reasoning**: Explains how conclusions were reached

## Usage Recommendations

1. **Use Batch Analysis**: Always analyze multiple videos together for better discrimination
2. **Look at Match Counts**: Number of matches is often more telling than similarity scores alone
3. **Trust the System**: The comparative analysis is more reliable than manual threshold adjustment
4. **Consider Context**: When in doubt, look at the detailed reasoning provided

## Example Output

```
Video Results:
┌─────────────────────────────────────────────┬────────────────┬──────────────┬──────────────┬──────────────┐
│ Video Name                                  │ Max Similarity │ Matches      │ Confidence   │ Likely Match │
├─────────────────────────────────────────────┼────────────────┼──────────────┼──────────────┼──────────────┤
│ Star_Trek_Voyager_S07e09-10_Flesh_And_Blood │ 83.97%         │ 3            │ High         │ YES          │
│ Star_Trek_Voyager_S07e24_Renaissance_Man    │ 79.59%         │ 0            │ Low          │ NO           │
└─────────────────────────────────────────────┴────────────────┴──────────────┴──────────────┴──────────────┘

Analysis Reasoning:
- Flesh and Blood has 3 matches vs 0 for Renaissance Man (infinite ratio)
- Flesh and Blood exceeds 80% confidence threshold
- 4.38% similarity difference supports the match count difference
```

This approach reliably distinguishes between similar confidence scores by using multiple metrics in combination.