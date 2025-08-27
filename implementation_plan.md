# Frame-Finder Analyzer Implementation Plan

## Overview

This document outlines the implementation plan for improving the frame-finder analyzer based on the technical specification. The improvements are organized by priority and dependencies to ensure a smooth implementation process.

## Phase 1: Critical Performance Improvements

### 1. GPU Acceleration and Model Upgrade
**Priority**: High
**Dependencies**: None

**Tasks**:
- [ ] Update analyzer.py to detect and use GPU if available
- [ ] Modify model loading to move model to GPU
- [ ] Update requirements.txt with appropriate torch version
- [ ] Test GPU utilization

**Implementation Details**:
```python
# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Move model to device
model = model.to(device)
```

### 2. Reference Embedding Caching
**Priority**: High
**Dependencies**: GPU Acceleration

**Tasks**:
- [ ] Create ReferenceEmbeddings class
- [ ] Modify compare_images function to use cached embeddings
- [ ] Update process_videos to use ReferenceEmbeddings
- [ ] Test caching performance improvement

### 3. Batch Frame Encoding
**Priority**: High
**Dependencies**: Reference Embedding Caching

**Tasks**:
- [ ] Modify extract_frames to support batch processing
- [ ] Update compare_images to handle batched inputs
- [ ] Implement batch_process_frames function
- [ ] Test batch processing performance

## Phase 2: Accuracy Improvements

### 4. Light Image Normalization
**Priority**: Medium
**Dependencies**: None

**Tasks**:
- [ ] Implement normalize_image function with CLAHE
- [ ] Apply normalization to both reference images and frames
- [ ] Add option to enable/disable normalization
- [ ] Test normalization impact on accuracy

### 5. Negative References + Delta Scoring
**Priority**: High
**Dependencies**: Reference Embedding Caching

**Tasks**:
- [ ] Add support for negative reference images in UI
- [ ] Modify Flask app to handle negative references
- [ ] Implement compute_delta_score function
- [ ] Update process_videos to use delta scoring
- [ ] Test false positive reduction

### 6. Temporal Clustering / Peak Picking
**Priority**: Medium
**Dependencies**: Negative References

**Tasks**:
- [ ] Implement cluster_detections function
- [ ] Add clustering parameter to UI
- [ ] Update results processing to apply clustering
- [ ] Test clustering effectiveness

## Phase 3: Advanced Features

### 7. Adaptive Thresholding
**Priority**: Medium
**Dependencies**: Negative References + Delta Scoring

**Tasks**:
- [ ] Implement compute_adaptive_threshold function
- [ ] Add option to enable adaptive thresholding
- [ ] Update process_videos to use adaptive thresholding
- [ ] Test threshold adaptation across different videos

### 8. Stronger Backbone Model
**Priority**: Low
**Dependencies**: GPU Acceleration

**Tasks**:
- [ ] Add model selection option to UI
- [ ] Implement support for CLIP-ViT-Large
- [ ] Implement support for SigLIP (optional)
- [ ] Test accuracy improvement
- [ ] Document VRAM requirements

### 9. Two-Stage Filter
**Priority**: Low
**Dependencies**: Light Image Normalization

**Tasks**:
- [ ] Implement two_stage_filter function
- [ ] Add option to enable two-stage filtering
- [ ] Update processing pipeline to use filter
- [ ] Test performance improvement

### 10. Micro-Tuning Around Peaks
**Priority**: Low
**Dependencies**: Temporal Clustering

**Tasks**:
- [ ] Implement micro_tune_around_peak function
- [ ] Add option to enable micro-tuning
- [ ] Update peak processing to apply micro-tuning
- [ ] Test precision improvement

## Phase 4: UI/UX Improvements

### 11. Flask Wiring Updates
**Priority**: High
**Dependencies**: Negative References Implementation

**Tasks**:
- [ ] Add negative reference upload field to index.html
- [ ] Modify app.py to handle negative references
- [ ] Update results display to show clustered results
- [ ] Test end-to-end negative reference workflow

### 12. Real-Time Output Viewer
**Priority**: Medium
**Dependencies**: None

**Tasks**:
- [ ] Implement WebSocket or Server-Sent Events in Flask
- [ ] Add progress display to index.html
- [ ] Update main.js to handle real-time updates
- [ ] Test real-time progress feedback

### 13. UI Updates
**Priority**: High
**Dependencies**: None

**Tasks**:
- [ ] Change default confidence interval to 75%
- [ ] Modify frame interval selector to support decimals
- [ ] Add negative reference upload field
- [ ] Add clustering options to UI
- [ ] Test all UI changes

## Implementation Order

1. **Week 1**: GPU Acceleration, Reference Embedding Caching
2. **Week 2**: Batch Frame Encoding, Light Image Normalization
3. **Week 3**: Negative References + Delta Scoring, Temporal Clustering
4. **Week 4**: Adaptive Thresholding, Flask Wiring Updates
5. **Week 5**: UI/UX Improvements, Real-Time Output Viewer
6. **Week 6**: Advanced Features (Two-Stage Filter, Micro-Tuning)
7. **Week 7**: Stronger Backbone Model, Final Testing
8. **Week 8**: Documentation and Optimization

## Testing Strategy

### Unit Tests
- [ ] ReferenceEmbeddings class
- [ ] Image normalization functions
- [ ] Delta scoring implementation
- [ ] Clustering algorithms
- [ ] Adaptive thresholding

### Integration Tests
- [ ] End-to-end processing pipeline
- [ ] GPU utilization
- [ ] Negative reference handling
- [ ] Real-time progress updates

### Performance Tests
- [ ] Processing time comparison (before/after)
- [ ] GPU utilization metrics
- [ ] Memory usage analysis
- [ ] Batch processing efficiency

### Accuracy Tests
- [ ] False positive reduction with negative references
- [ ] Precision improvement with clustering
- [ ] Threshold adaptation effectiveness
- [ ] Model accuracy comparison

## Risk Mitigation

1. **GPU Memory Issues**:
   - Implement automatic fallback to CPU
   - Add batch size adjustment based on available VRAM
   - Monitor memory usage during processing

2. **Performance Degradation**:
   - Maintain backward compatibility
   - Provide options to disable new features
   - Benchmark each improvement

3. **UI Compatibility**:
   - Test on multiple browsers
   - Ensure mobile responsiveness
   - Provide clear user guidance

## Success Metrics

1. **Performance**:
   - 50% reduction in processing time
   - 90% GPU utilization when available
   - Support for batch processing of 32+ frames

2. **Accuracy**:
   - 30% reduction in false positives
   - 20% improvement in true positive rate
   - Adaptive thresholding effectiveness > 80%

3. **User Experience**:
   - Real-time progress updates
   - Intuitive UI for new features
   - 95% user satisfaction rating

## Documentation Updates

1. **User Guide**:
   - New feature explanations
   - Updated UI walkthrough
   - Performance optimization tips

2. **Technical Documentation**:
   - API documentation for new functions
   - Architecture diagrams
   - Implementation details

3. **README Updates**:
   - New requirements
   - Installation instructions
   - Usage examples