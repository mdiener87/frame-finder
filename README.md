# Frame Finder

A tool to detect reference images in video files using multiple approaches for improved accuracy.

## Overview

This project implements a robust frame detection system that can:
- Identify when a reference image appears in videos
- Minimize false positives through statistical analysis
- Provide confidence scores for detections
- Automatically determine which videos in a batch are most likely to contain the reference

## Why This Problem Is Challenging

The task of detecting specific objects in videos is more complex than it initially appears:

1. **Object Detection vs. Image Classification**: CLIP excels at semantic understanding ("what is this?") but struggles with precise object matching ("where is this?").

2. **Scale and Context Issues**: Objects that are small in the frame or appear in different contexts are harder to detect reliably.

3. **Visual Similarity Ambiguity**: Models can sometimes find "false" similarities based on color, texture, or composition rather than actual object matching.

4. **Confidence Calibration**: Model confidence scores are not always well-calibrated probabilities.

## Approaches Implemented

### 1. Basic CLIP-based Detection (`frame_analyzer.py`)
- Uses OpenAI's CLIP model for semantic similarity
- Implements statistical filtering to reduce false positives
- Works well for clear, unambiguous matches

### 2. Enhanced CLIP with Preprocessing (`enhanced_analyzer.py`)
- Applies image normalization (CLAHE) to handle lighting variations
- Uses larger CLIP models for better accuracy
- Implements differential similarity with negative samples

### 3. Multi-Strategy Approach (`multi_strategy_analyzer.py`)
- Combines CLIP semantic similarity with traditional computer vision (ORB features)
- Uses weighted combination of multiple similarity measures
- More robust for challenging detection scenarios

### 4. Strict Analysis with Comparative Logic (`strict_analyzer.py`)
- Uses multiple validation criteria that all must be met
- Implements background comparison for better context
- Provides clear differentiation between positive and negative results

## Unified Web Interface

The project includes a single, unified web interface:

### Main Page (`/`)
- Landing page with link to start analysis

### Video Analysis (`/analyze`)
- Upload a reference image and one or multiple videos
- Get automatic analysis showing which videos contain the reference
- See detailed results including:
  - Which videos likely contain the reference
  - Top match timestamps for each video
  - Confidence levels for each determination
  - Detailed reasoning for each result

## API Endpoints

### POST /api/analyze
Start analysis of one or more videos.

Form Data:
- `reference`: Reference image file
- `videos`: One or more video files

Response:
```json
{
  "analysis_id": "unique-id",
  "status": "processing",
  "message": "Analysis started"
}
```

### GET /api/analysis/<analysis_id>
Get results of analysis.

Response:
```json
{
  "status": "completed",
  "results": {
    "video1.mp4": {
      "found": true,
      "confidence": 0.86,
      "max_similarity": 0.86,
      "matches": [...],
      "video_name": "video1.mp4"
    }
  },
  "comparison": {
    "video1.mp4": {
      "is_likely_match": true,
      "confidence_level": "high",
      "max_similarity": 0.86,
      "matches_count": 5,
      "reasoning": "Strong match: 5 matches, 0.860 similarity"
    }
  }
}
```

## Running Tests

```bash
python test_frame_analyzer.py
```

## Running the Web Interface

```bash
python api.py
```

The web interface will be available at http://localhost:5000

## Key Features

- **Unified Interface**: Single interface for both single and multiple video analysis
- **Automatic Determination**: System automatically determines which videos contain the reference
- **Detailed Results**: Shows which videos likely contain the reference, top match timestamps, and confidence levels
- **Comparative Analysis**: When analyzing multiple videos, the system compares them to determine which are most likely matches
- **No Manual Threshold Adjustment**: The system automatically determines likely matches without requiring manual threshold adjustment