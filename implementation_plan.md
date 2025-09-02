# Implementation Plan for New Prop Detection Algorithm

## Overview

This document outlines the step-by-step implementation plan for the new prop detection algorithm based on the technical specification. The implementation will be done in phases to ensure proper testing and integration.

## Phase 1: Core Algorithm Implementation

### 1. Preprocessing Pipeline

#### 1.1 Reference Image Preprocessing
- Implement image loading and format conversion
- Add tight crop functionality (user responsibility, but provide utilities)
- Implement gamma correction and normalization to [0,1] range
- Add letterbox/pad and center-crop functionality
- Create grayscale conversion utility
- Implement caching mechanism for processed references

#### 1.2 Video Frame Preprocessing
- Implement consistent preprocessing with reference images
- Add aspect handling utilities
- Implement optional CLAHE normalization

### 2. Multi-scale Template Matching

#### 2.1 Template Pyramid Generation
- Implement scaling functions to create template pyramid (0.6–1.6× in ~10 steps)
- Add utility functions for template scaling

#### 2.2 Normalized Cross-Correlation
- Implement OpenCV-based NCC matching
- Add peak detection for top-K matches per scale
- Implement bounding box conversion from matches

### 3. LPIPS Verification

#### 3.1 LPIPS Network Integration
- Integrate LPIPS library with PyTorch backend
- Implement network initialization and caching
- Add GPU support for LPIPS computations

#### 3.2 Candidate Verification
- Implement LPIPS distance calculation for candidate crops
- Add thresholding mechanism (LPIPS < T_lpips)

### 4. CLIP Verification

#### 4.1 CLIP Model Integration
- Integrate open-clip-torch library
- Implement CLIP embedding computation
- Add GPU support for CLIP computations

#### 4.2 Cosine Similarity Calculation
- Implement cosine similarity computation between embeddings
- Add thresholding mechanism (cosine > T_clip)

## Phase 2: Advanced Features

### 1. Temporal Smoothing and Deduplication

#### 1.1 Non-Maximum Suppression
- Implement per-frame NMS with IoU threshold (≥ 0.5)
- Add bounding box merging functionality

#### 1.2 Track Consistency
- Implement centroid IoU tracking frame-to-frame
- Add tracking utilities

#### 1.3 Debounce Mechanism
- Implement N of M consecutive frame requirement (e.g., ≥3 of 12)
- Add event emission logic

#### 1.4 Cool-down Period
- Implement re-trigger suppression with spatial movement checking

### 2. Calibration and Thresholding

#### 2.1 Calibration Set Generation
- Implement utility for building calibration sets
- Add positive/negative sample handling

#### 2.2 Threshold Selection
- Implement ROC/PR curve generation using scikit-learn
- Add threshold optimization utilities

### 3. Optional Features

#### 3.1 ORB Confirmation
- Implement ORB keypoint detection and matching
- Add in-box inlier ratio calculation

#### 3.2 Score Fusion
- Implement z-score computation for LPIPS and CLIP scores
- Add score fusion mechanisms (sum or minimum)

## Phase 3: Performance Optimizations

### 1. Frame Stride Implementation
- Add configurable frame stride parameter
- Implement back-fill logic around hits

### 2. Resolution Handling
- Add downscaling utilities for proposal stage
- Implement resolution management for verification stage

### 3. GPU Utilization
- Ensure LPIPS & CLIP run on CUDA when available
- Implement parallelization for proposals and verifications

## Phase 4: Integration and Testing

### 1. Backend Integration
- Replace current analyzer.py with new implementation
- Maintain API compatibility where possible
- Add new configuration options

### 2. Frontend Integration
- Update frontend to work with new backend
- Add any new UI elements if needed
- Maintain existing user experience

### 3. Testing
- Implement unit tests for each component
- Add integration tests for the full pipeline
- Create performance benchmarks
- Validate accuracy improvements

## Detailed Component Implementation Order

### Week 1: Foundation
1. Preprocessing pipeline (reference images and video frames)
2. Template pyramid generation
3. Basic NCC matching implementation
4. Initial LPIPS integration

### Week 2: Core Verification
1. Complete LPIPS verification with thresholding
2. CLIP model integration
3. CLIP cosine similarity implementation
4. Basic score fusion

### Week 3: Temporal Processing
1. Non-Maximum Suppression implementation
2. Track consistency mechanisms
3. Debounce and cool-down implementation

### Week 4: Calibration and Optimization
1. Calibration set generation utilities
2. Threshold selection with ROC/PR curves
3. Performance optimizations (frame stride, resolution handling)

### Week 5: Advanced Features
1. ORB confirmation implementation
2. Enhanced score fusion
3. GPU utilization improvements

### Week 6: Integration
1. Backend API updates
2. Frontend compatibility
3. Parameter tuning and validation

### Week 7: Testing and Documentation
1. Unit tests for all components
2. Integration testing
3. Performance benchmarking
4. Documentation updates

## Risk Mitigation

### Technical Risks
1. **GPU Memory Issues**: Implement fallback to CPU processing
2. **Performance Bottlenecks**: Profile each component and optimize critical paths
3. **Accuracy Issues**: Extensive calibration and threshold tuning

### Mitigation Strategies
1. Implement progressive processing with early exits
2. Add detailed logging for debugging
3. Create comprehensive test suite with known good examples
4. Provide clear error messages and failure handling

## Success Metrics

### Accuracy
- Precision ≥ 95% on test set
- Recall ≥ 80% on test set
- Detections persist ≥ 3 frames

### Performance
- Processing time for 2-hour video < 30 minutes on mid-range hardware
- Memory usage < 8GB during processing
- GPU utilization > 70% when available

### Usability
- Same or improved user experience
- Clear error messages and progress reporting
- Comprehensive documentation