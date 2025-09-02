# Frame Finder - New Prop Detection Algorithm Project Roadmap

## Project Overview

This document provides a comprehensive roadmap for implementing a completely redesigned prop detection algorithm for Frame Finder. The new approach follows the "One-Shot Prop Finder" implementation outline, focusing on high-confidence detection of film props in video frames with improved accuracy and performance.

## Completed Architectural Planning

### 1. Technical Specification
- Defined the three-stage algorithm architecture:
  1. Preprocessing pipeline
  2. Candidate proposal stage (multi-scale template matching)
  3. Candidate verification stage (LPIPS + CLIP)
  4. Temporal smoothing and deduplication

### 2. Dependencies Analysis
- Identified required additional dependencies:
 - lpips (Perceptual similarity metrics)
  - open-clip-torch (CLIP embeddings)
  - scikit-learn (ROC/PR curves for threshold calibration)

### 3. Implementation Plan
- Created a detailed 7-week implementation plan:
  - Week 1: Foundation (preprocessing, template matching)
  - Week 2: Core verification (LPIPS, CLIP)
  - Week 3: Temporal processing
  - Week 4: Calibration and optimization
  - Week 5: Advanced features
  - Week 6: Integration
  - Week 7: Testing and documentation

### 4. Backend API Update Plan
- Designed new processing function with enhanced parameters:
  - Frame processing controls (stride, resolution)
  - Verification thresholds (LPIPS, CLIP)
  - Temporal parameters (NMS, debounce)
- Maintained API compatibility where possible while allowing breaking changes

### 5. Frontend Integration Plan
- Designed enhanced UI with:
  - Advanced settings panel
  - New configuration options
  - Preset buttons for common use cases
 - Enhanced results display with individual scores
  - Processing statistics dashboard

### 6. Testing Strategy
- Comprehensive testing approach covering:
  - Unit testing for all components
  - Integration testing for API and frontend
  - Performance testing for speed and resource usage
 - Validation testing for accuracy metrics
  - Edge case and regression testing

## Implementation Roadmap

### Phase 1: Core Algorithm Implementation (Weeks 1-2)
**Objective**: Implement the foundational components of the new algorithm

**Deliverables**:
- Preprocessing pipeline for reference images and video frames
- Multi-scale template matching implementation
- LPIPS verification with thresholding
- CLIP verification with cosine similarity

**Success Criteria**:
- Basic prop detection working with test data
- LPIPS and CLIP scores computed correctly
- Template matching producing reasonable proposals

### Phase 2: Advanced Features (Weeks 3-4)
**Objective**: Implement temporal processing and calibration mechanisms

**Deliverables**:
- Non-Maximum Suppression implementation
- Track consistency mechanisms
- Debounce and cool-down implementation
- Calibration set generation utilities
- Threshold selection with ROC/PR curves

**Success Criteria**:
- Temporal smoothing reducing duplicate detections
- Calibration producing optimal thresholds
- Improved accuracy over baseline implementation

### Phase 3: Performance Optimizations (Week 5)
**Objective**: Optimize performance and implement advanced features

**Deliverables**:
- Frame stride implementation
- Resolution handling optimizations
- GPU utilization improvements
- ORB confirmation implementation (optional)
- Enhanced score fusion mechanisms

**Success Criteria**:
- Processing time reduced by 30% compared to baseline
- GPU utilization > 70% when available
- Memory usage within specified limits

### Phase 4: Integration (Week 6)
**Objective**: Integrate new algorithm with existing frontend and backend

**Deliverables**:
- Backend API updates with new parameters
- Frontend UI enhancements
- Parameter tuning and validation
- Documentation updates

**Success Criteria**:
- Seamless integration with existing workflow
- Enhanced UI providing access to new features
- Backward compatibility maintained where appropriate

### Phase 5: Testing and Validation (Week 7)
**Objective**: Comprehensive testing and validation of the new implementation

**Deliverables**:
- Unit tests for all components
- Integration tests for full pipeline
- Performance benchmarking
- Accuracy validation against test sets
- User experience testing

**Success Criteria**:
- Precision ≥ 95% and Recall ≥ 80% on validation set
- Processing time < 30 minutes for 2-hour video
- Positive feedback from user testing
- Zero critical issues in production

## Key Features of the New Implementation

### Improved Accuracy
- Multi-stage verification using both LPIPS and CLIP
- Temporal consistency checking to reduce false positives
- Calibration-based threshold optimization
- Optional ORB confirmation for challenging cases

### Enhanced Performance
- Frame stride processing for faster analysis
- GPU acceleration for similarity computations
- Resolution optimization for different processing stages
- Parallelization of proposals and verifications

### Better User Experience
- Advanced configuration options with presets
- Enhanced results display with individual scores
- Processing statistics and performance metrics
- Improved progress reporting and error handling

### Robustness
- Better handling of lighting variations
- Improved robustness to rotation/scale changes
- Edge case handling for challenging video conditions
- Graceful degradation when GPU is not available

## Risk Mitigation

### Technical Risks
1. **Accuracy Issues**
   - Mitigation: Extensive calibration and validation
   - Contingency: Parameter tuning and threshold adjustment

2. **Performance Bottlenecks**
   - Mitigation: Profiling and optimization
   - Contingency: Fallback to CPU processing

3. **Integration Issues**
   - Mitigation: Comprehensive integration testing
   - Contingency: Rollback to previous implementation

### Project Risks
1. **Timeline Delays**
   - Mitigation: Weekly progress reviews
   - Contingency: Prioritization of core features

2. **Resource Constraints**
   - Mitigation: Cloud-based testing resources
   - Contingency: Focused testing on critical components

## Success Metrics

### Accuracy Goals
- Precision: ≥ 95%
- Recall: ≥ 80%
- F1-Score: ≥ 85%

### Performance Goals
- Processing time for 2-hour video: < 30 minutes (high-end system)
- Memory usage: < 8GB during processing
- GPU utilization: > 70% when available

### User Experience Goals
- Positive feedback from beta testing
- Task completion time reduced by 20%
- Zero critical usability issues

## Next Steps

1. **Switch to Code Mode**: Begin implementation of the core algorithm components
2. **Environment Setup**: Install additional dependencies (lpips, open-clip-torch)
3. **Phase 1 Implementation**: Start with preprocessing pipeline and template matching
4. **Continuous Testing**: Implement unit tests alongside development
5. **Weekly Reviews**: Assess progress against timeline and adjust as needed

This roadmap provides a clear path to implementing the new prop detection algorithm while ensuring quality, performance, and user satisfaction.