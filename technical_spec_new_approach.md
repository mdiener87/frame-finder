# Frame Finder - New Prop Detection Algorithm Technical Specification

## Overview

This document outlines the technical specification for a completely redesigned prop detection algorithm for Frame Finder. The new approach follows the "One-Shot Prop Finder" implementation outline from ChatGPT, focusing on high-confidence detection of film props in video frames.

## Algorithm Architecture

The new algorithm consists of three main stages:

1. **Preprocessing**: Normalization and preparation of reference images and video frames
2. **Candidate Proposals**: Identification of potential prop locations using multi-scale template matching
3. **Candidate Verification**: Confirmation of candidates using LPIPS and CLIP similarity metrics
4. **Temporal Smoothing**: Deduplication and temporal consistency checking

## 1. Preprocessing Pipeline

### Reference Image Preprocessing
- Tight crop the prop (user responsibility)
- Optional binary mask creation to isolate the prop from background
- Convert to RGB, float32, [0,1] range
- Apply gamma correction for consistent lighting
- Letterbox/pad to target size then center-crop for CLIP/LPIPS
- Create grayscale copy for NCC/ORB
- Cache:
  - Masked reference (RGB + gray)
  - CLIP embedding of the reference (L2-normalized)
  - LPIPS network ready

### Video Frame Preprocessing
- Consistent preprocessing with reference images
- Aspect handling: letterbox/pad to target size then center-crop
- Grayscale conversion for template matching
- Optional light color-jitter normalization (histogram equalization or CLAHE)

## 2. Candidate Proposal Stage

### Multi-scale Template Matching (Primary Method)
- Use OpenCV Normalized Cross-Correlation (cv2.matchTemplate)
- Build pyramid of down/up-scaled templates (0.6–1.6× in ~10 steps)
- Output top-K peaks per scale (score, box)
- Pros: Simple, fast, great for near-frontal, limited rotation
- Cons: Sensitive to rotation/occlusion

### Keypoint Geometric Match (Alternative/Supplementary)
- ORB (or SIFT if license permits) + ratio test + RANSAC homography
- Output: Inlier count/ratio + projected quad → convert to bounding box
- Use when the prop has texture/edges; handles mild rotation/scale

## 3. Candidate Verification Stage

### Primary Verification: LPIPS Distance
- Apply to each candidate crop (with mask if available)
- Preprocess crop and reference identically (size, color)
- Decision threshold: LPIPS < T_lpips (suggested: 0.25–0.40)

### Secondary Verification: CLIP Cosine Similarity
- L2-normalize embeddings
- Compute cosine similarity
- Decision threshold: cosine > T_clip (suggested: 0.28–0.38)

### Optional Confirmation: ORB In-box Inlier Ratio
- Compute ORB in-box inlier ratio to reject busy-background false positives
- Decision threshold: > r_min (tune 14–24 depending on clip/ref)

### Score Fusion
- Compute z-scores for LPIPS (negated) and CLIP cosine from calibration set
- Sum or take minimum for conservative gating

## 4. Temporal Smoothing & Deduplication

### Per-frame NMS
- Merge overlapping boxes (IoU ≥ 0.5) keeping best verified score

### Track Consistency
- Simple centroid IoU tracking frame-to-frame

### Debounce Mechanism
- Require N of M consecutive frames over threshold (e.g., ≥3 of 12) before emitting a detection event

### Cool-down Period
- Suppress re-triggers within a window unless the box moves significantly

## 5. Calibration & Thresholds

### Calibration Set
- Build ~30–80 candidate crops:
  - Half positives (selected frames where prop visibly appears)
  - Half hard negatives (similar scenes without the prop)

### Threshold Selection
- Run Stage A→B on calibration set
- Collect LPIPS and CLIP scores
- Use ROC/PR (scikit-learn) to pick operating points
- Fix thresholds globally; no per-batch/per-video scaling

## 6. Performance Optimizations

### Frame Stride
- Analyze every Nth frame (e.g., 2–3) then back-fill around hits

### Resolution Handling
- Downscale long side to ~720–1080px for proposals
- Re-verify at native or half-res

### GPU Utilization
- LPIPS & CLIP on CUDA
- Template/ORB on CPU
- Parallelize proposals and verifications with a small worker pool

## 7. Output & QA

### Debug Information
- Save debug overlays (boxes + LPIPS/CLIP scores) for sampled frames and hits

### Per-video Reporting
- Number of proposals
- Verified hits
- Final events
- Precision/Recall on calibration set

### Acceptance Criteria
- Precision ≥ 95%
- Recall ≥ 80% on held-out clips
- Detections persist ≥ 3 frames

## 8. Dependencies

### Core Libraries
- opencv-python (I/O, resize, NCC, ORB/SIFT, NMS)
- numpy
- ffmpeg-python (frame extract)
- torch (LPIPS/CLIP)
- scikit-learn (ROC/PR)
- matplotlib (optional debug plots)

### Similarity Libraries
- lpips (Perceptual)
- open-clip-torch (CLIP embeddings)
- kornia (image ops, normalization)

### Optional Libraries
- scikit-image (SSIM)
- rich/typer (CLI & logs)

## 9. Implementation Plan

### Phase 1: Core Algorithm Implementation
1. Implement preprocessing pipeline
2. Implement multi-scale template matching
3. Implement LPIPS verification
4. Implement CLIP verification

### Phase 2: Advanced Features
1. Implement temporal smoothing and deduplication
2. Implement calibration and thresholding
3. Add optional ORB confirmation
4. Add performance optimizations

### Phase 3: Integration
1. Replace current analyzer.py with new implementation
2. Update API endpoints if needed
3. Maintain frontend compatibility
4. Add comprehensive testing

## 10. Expected Improvements

### Accuracy
- Higher confidence for positive matches
- Lower confidence for frames without the image
- Better handling of lighting variations
- Improved robustness to rotation/scale changes

### Performance
- More efficient processing with frame stride
- GPU acceleration for similarity computations
- Better temporal consistency

### Usability
- Simplified user workflow with better defaults
- More reliable results with fewer false positives
- Better diagnostics and debugging information