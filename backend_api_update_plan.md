# Backend API Update Plan for New Prop Detection Algorithm

## Overview

This document outlines the changes needed to update the backend API to support the new prop detection algorithm. Since we have permission to make breaking changes, we'll design an improved API that better supports the new functionality.

## Current API Analysis

### Existing Endpoints
1. `/` - Main upload page (GET)
2. `/upload` - Handle file uploads and start processing (POST)
3. `/task_status/<task_id>` - Check processing task status (GET)
4. `/results/<task_id>` - Display results for completed task (GET)
5. `/cancel_task/<task_id>` - Cancel a processing task (POST)
6. `/health` - Health check endpoint (GET)
7. `/export_results/<task_id>` - Export results as JSON (POST)

### Current Processing Function
The current system uses `process_videos` function in `analyzer.py` with the following signature:
```python
def process_videos(reference_paths, video_paths, frame_interval=1.0,
                   confidence_threshold=0.5, negative_paths=None, progress_callback=None):
```

## Proposed New API Design

### Core Processing Function
We'll replace the current `process_videos` with a new implementation that supports the three-stage approach:

```python
def process_videos_new(reference_paths, video_paths, negative_paths=None, 
                       frame_interval=1.0, frame_stride=1, resolution_target=1080,
                       lpips_threshold=0.35, clip_threshold=0.33,
                       nms_iou_threshold=0.5, debounce_n=3, debounce_m=12,
                       progress_callback=None):
    """
    Process videos using the new three-stage prop detection algorithm.
    
    Args:
        reference_paths (list[str]): Paths to positive reference images
        video_paths (list[str]): Paths to video files
        negative_paths (list[str], optional): Paths to negative reference images
        frame_interval (float): Seconds between frames to process
        frame_stride (int): Process every Nth frame then back-fill around hits
        resolution_target (int): Target resolution for processing (longest side)
        lpips_threshold (float): LPIPS distance threshold for verification
        clip_threshold (float): CLIP cosine similarity threshold for verification
        nms_iou_threshold (float): IoU threshold for non-maximum suppression
        debounce_n (int): Minimum consecutive frames for detection
        debounce_m (int): Window size for consecutive frame checking
        progress_callback (callable): Callback for progress reporting
        
    Returns:
        dict: Processing results with timestamps, confidence scores, and metadata
    """
```

### Configuration Options

#### Preprocessing Parameters
- `frame_interval`: Seconds between frames to process (default: 1.0)
- `frame_stride`: Process every Nth frame then back-fill (default: 1)
- `resolution_target`: Target resolution for processing (default: 1080)

#### Verification Thresholds
- `lpips_threshold`: LPIPS distance threshold (default: 0.35)
- `clip_threshold`: CLIP cosine similarity threshold (default: 0.33)

#### Temporal Parameters
- `nms_iou_threshold`: IoU threshold for non-maximum suppression (default: 0.5)
- `debounce_n`: Minimum consecutive frames for detection (default: 3)
- `debounce_m`: Window size for consecutive frame checking (default: 12)

### API Endpoint Changes

#### 1. `/upload` Endpoint
The upload endpoint will need to support additional parameters:

```python
# New form parameters to support:
# - frame_interval (existing, keep as is)
# - frame_stride (new)
# - resolution_target (new)
# - lpips_threshold (new)
# - clip_threshold (new)
# - nms_iou_threshold (new)
# - debounce_n (new)
# - debounce_m (new)
```

#### 2. `/task_status/<task_id>` Endpoint
This endpoint can remain largely the same, but we might want to add more detailed progress information:

```python
# Enhanced status information:
{
    'status': 'processing',
    'progress': 0-100,
    'current_video': 'filename.mp4',
    'current_stage': 'preprocessing|proposal|verification|smoothing',
    'stage_progress': 0-100,
    'results': None,
    'error': None
}
```

#### 3. `/results/<task_id>` Endpoint
The results structure will change to reflect the new algorithm's output:

```python
# New results structure:
{
    'video_name.mp4': {
        'matches': [
            {
                'timestamp': 123.456,
                'confidence': 0.95,
                'lpips_score': 0.25,
                'clip_score': 0.85,
                'bounding_box': [x, y, width, height],  # New field
                'reference_image': 'ref1.jpg'
            }
        ],
        'max_confidence': 0.95,
        'thresholds_used': {
            'lpips': 0.35,
            'clip': 0.33
        },
        'processing_stats': {
            'total_frames_processed': 1250,
            'total_proposals': 5420,
            'total_verifications': 320,
            'final_detections': 15
        }
    }
}
```

## Integration with Existing Code

### App.py Changes
1. Update the `process_videos_background` function to use the new algorithm
2. Modify parameter extraction in `/upload` endpoint to support new parameters
3. Update progress callback to report stage information
4. Maintain backward compatibility where possible

### Analyzer.py Replacement
The entire `analyzer.py` file will be replaced with the new implementation:

1. Remove all existing code
2. Implement the new three-stage algorithm:
   - Preprocessing pipeline
   - Candidate proposal stage
   - Candidate verification stage
   - Temporal smoothing stage
3. Maintain the same function signature for `process_videos` but with new implementation
4. Add new utility functions as needed

## Backward Compatibility

While we're allowed to make breaking changes, we should maintain the same basic workflow:

1. Users still upload reference images and videos
2. Processing still happens in the background
3. Results are still displayed in the same format (with enhancements)
4. Export functionality still works (with enhanced data)

## New Features to Expose

### Enhanced Results Display
- Show bounding boxes on detected frames
- Display LPIPS and CLIP scores separately
- Show processing statistics
- Add confidence filtering based on individual scores

### Advanced Configuration
- Allow users to adjust algorithm parameters through the UI
- Provide presets for different use cases (high precision, high recall, balanced)

## Implementation Steps

### Step 1: Core Algorithm Implementation
1. Create new analyzer module with basic preprocessing
2. Implement multi-scale template matching
3. Add LPIPS verification
4. Add CLIP verification

### Step 2: Advanced Features
1. Implement temporal smoothing
2. Add calibration utilities
3. Implement ORB confirmation (optional)
4. Add performance optimizations

### Step 3: API Integration
1. Update app.py to use new analyzer
2. Modify endpoints to support new parameters
3. Update progress reporting
4. Maintain result format compatibility

### Step 4: Testing and Validation
1. Test with existing data sets
2. Validate accuracy improvements
3. Ensure performance meets requirements
4. Update documentation