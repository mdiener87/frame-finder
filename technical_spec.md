# Frame-Finder Analyzer Technical Specification

## Current Implementation Analysis

The current analyzer.py implementation has several areas for improvement:

1. **Performance Issues**:
   - Reference embeddings are computed for each frame comparison
   - No GPU acceleration is utilized
   - No batching of frame processing
   - No caching of reference embeddings

2. **Accuracy Issues**:
   - No negative reference support
   - Fixed confidence threshold
   - No temporal clustering
   - No adaptive thresholding

3. **UI/UX Issues**:
   - No real-time progress feedback
   - Frame interval only supports integers
   - Fixed 50% confidence threshold

## Improvement Plan

### 1. Cache Reference Embeddings

**Problem**: Reference embeddings are computed for each frame comparison, causing redundant computation.

**Solution**: 
- Compute reference embeddings once at startup
- Cache normalized embedding vectors
- Reuse cached embeddings for all frame comparisons

**Implementation**:
```python
class ReferenceEmbeddings:
    def __init__(self, reference_paths, model, processor, device):
        self.embeddings = []
        self.paths = reference_paths
        self.model = model
        self.processor = processor
        self.device = device
        self._compute_embeddings()
    
    def _compute_embeddings(self):
        for path in self.paths:
            image = Image.open(path)
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                self.embeddings.append(features.cpu())
```

### 2. Light Image Normalization

**Problem**: Inconsistent lighting and compression artifacts affect similarity scores.

**Solution**: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel of LAB color space.

**Implementation**:
```python
def normalize_image(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    lab = cv2.merge((l, a, b))
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(normalized)
```

### 3. Negative References + Delta Scoring

**Problem**: High false positive rate due to hull/background matches.

**Solution**: 
- Add support for negative reference images
- Compute delta score: max(sim(frame, positives)) - max(sim(frame, negatives))

**Implementation**:
```python
def compute_delta_score(frame_features, positive_embeddings, negative_embeddings):
    # Compute similarities with positive references
    positive_scores = []
    for emb in positive_embeddings:
        similarity = torch.nn.functional.cosine_similarity(frame_features, emb).item()
        positive_scores.append(similarity)
    
    # Compute similarities with negative references
    negative_scores = []
    if negative_embeddings:
        for emb in negative_embeddings:
            similarity = torch.nn.functional.cosine_similarity(frame_features, emb).item()
            negative_scores.append(similarity)
    
    # Compute delta score
    max_positive = max(positive_scores) if positive_scores else 0
    max_negative = max(negative_scores) if negative_scores else 0
    
    return max_positive - max_negative
```

### 4. Batch Frame Encoding

**Problem**: Processing frames one by one is inefficient and doesn't leverage GPU parallelization.

**Solution**: Process frames in batches to improve throughput and CLIP signal.

**Implementation**:
```python
def batch_process_frames(frames, batch_size=32):
    results = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        # Process batch
        batch_results = process_batch(batch)
        results.extend(batch_results)
    return results
```

### 5. Temporal Clustering / Peak Picking

**Problem**: Multiple detections of the same event due to wobble or similar frames.

**Solution**: Cluster hits within ~±1s and keep the highest-scoring frame per cluster.

**Implementation**:
```python
def cluster_detections(detections, time_window=1.0):
    # Sort detections by timestamp
    detections.sort(key=lambda x: x['timestamp'])
    
    clusters = []
    current_cluster = []
    
    for detection in detections:
        if not current_cluster:
            current_cluster.append(detection)
        else:
            # Check if within time window
            if detection['timestamp'] - current_cluster[0]['timestamp'] <= time_window:
                current_cluster.append(detection)
            else:
                # Close current cluster and start new one
                clusters.append(max(current_cluster, key=lambda x: x['confidence']))
                current_cluster = [detection]
    
    # Don't forget the last cluster
    if current_cluster:
        clusters.append(max(current_cluster, key=lambda x: x['confidence']))
    
    return clusters
```

### 6. Adaptive Thresholding

**Problem**: Hardcoded 0.5 threshold doesn't work well across different videos.

**Solution**: Sample background frames, compute delta-scores, set threshold to μ + 3σ or 99.5th percentile.

**Implementation**:
```python
def compute_adaptive_threshold(background_frames, reference_embeddings, negative_embeddings):
    scores = []
    for frame in background_frames:
        # Process frame and compute score
        score = compute_delta_score(frame, reference_embeddings, negative_embeddings)
        scores.append(score)
    
    # Compute threshold as 99.5th percentile
    if scores:
        threshold = np.percentile(scores, 99.5)
        return max(threshold, 0.1)  # Minimum threshold of 0.1
    
    return 0.5  # Default fallback
```

### 7. Stronger Backbone Model

**Problem**: Current CLIP-ViT-Base model may not provide sufficient visual discrimination.

**Solution**: Upgrade to CLIP-ViT-Large or SigLIP for finer visual discrimination.

**Implementation Options**:
1. `openai/clip-vit-large-patch14`
2. `google/siglip-base-patch16-256`

### 8. Two-Stage Filter

**Problem**: CLIP processing is computationally expensive for all frames.

**Solution**: Use cheap OpenCV gate (NCC or ORB/AKAZE match count) → CLIP re-check only for candidates.

**Implementation**:
```python
def two_stage_filter(frame, reference_images, threshold=0.7):
    # Stage 1: OpenCV template matching
    frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    
    for ref_img in reference_images:
        ref_cv = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
        result = cv2.matchTemplate(frame_cv, ref_cv, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > threshold:
            return True  # Pass to Stage 2 (CLIP)
    
    return False  # Reject early
```

### 9. Micro-Tuning Around Peaks

**Problem**: Initial detection may not be at the optimal frame.

**Solution**: When candidate detected, rescan ±2s at higher FPS (e.g., 4 fps) and keep max score.

**Implementation**:
```python
def micro_tune_around_peak(video_path, peak_timestamp, window=2.0, high_fps=4):
    # Extract frames at higher FPS around the peak
    start_time = max(0, peak_timestamp - window)
    end_time = peak_timestamp + window
    
    high_res_frames = extract_frames_high_res(video_path, start_time, end_time, 1/high_fps)
    
    # Find the frame with highest score in this window
    best_frame = None
    best_score = -1
    
    for frame, timestamp in high_res_frames:
        score = compute_similarity(frame, reference_embeddings)
        if score > best_score:
            best_score = score
            best_frame = (frame, timestamp)
    
    return best_frame
```

### 10. Flask Wiring Updates

**Problem**: Current Flask app doesn't support negative references or new features.

**Solution**: 
- Allow uploading "negative refs"
- Expose clustered/peak-picked results with adaptive threshold

**Implementation**:
- Add new file upload field for negative references
- Modify processing pipeline to handle negative references
- Update results display to show clustered results

### 11. Real-Time Output Viewer

**Problem**: No feedback during analysis process.

**Solution**: Add real-time progress updates to the web page.

**Implementation**:
- Add WebSocket or Server-Sent Events for real-time updates
- Show current video being processed
- Show progress percentage
- Show current matches found

### 12. UI Improvements

**Changes Needed**:
1. Change default confidence interval to 75%
2. Support decimal values in frame interval selector
3. Add negative reference upload field
4. Add real-time progress display

## Implementation Priority

1. **Critical Performance Improvements**:
   - Cache reference embeddings
   - GPU acceleration
   - Batch frame encoding

2. **Accuracy Improvements**:
   - Light image normalization
   - Negative references + delta scoring
   - Temporal clustering

3. **Advanced Features**:
   - Adaptive thresholding
   - Stronger backbone model
   - Two-stage filter

4. **UI/UX Improvements**:
   - Real-time output viewer
   - UI updates for new features

## Dependencies

- PyTorch with CUDA support
- OpenCV with contrib modules (for advanced features)
- Updated requirements.txt with new dependencies

## Testing Plan

1. Unit tests for each new component
2. Integration tests for the full pipeline
3. Performance benchmarks
4. Accuracy validation on sample datasets