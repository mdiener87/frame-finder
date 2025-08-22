# Frame-Finder UI Design Specification

## Overview

This document outlines the UI design changes needed to support the new analyzer features. The design focuses on maintaining usability while adding powerful new functionality.

## Current UI Analysis

The current UI has the following components:
- Reference image upload
- Video file upload
- Frame interval selector (integer only)
- Confidence threshold slider (0-100%)
- Results display table

## Proposed UI Improvements

### 1. Negative Reference Images

**Location**: Below the existing reference images section

**Design**:
```html
<div class="mb-3">
    <label for="negative_references" class="form-label">Negative Reference Images</label>
    <input class="form-control" type="file" id="negative_references" name="negative_references" multiple accept=".png,.jpg,.jpeg">
    <div class="form-text">Upload images that should NOT be detected (e.g., hull/background images)</div>
    <div id="negativePreview" class="mt-2"></div>
</div>
```

**Functionality**:
- Preview thumbnails of negative references
- Support multiple file selection
- Same file type restrictions as positive references

### 2. Advanced Processing Options

**Location**: Below the confidence threshold section

**Design**:
```html
<div class="card mt-4">
    <div class="card-header">
        <h5 class="mb-0">
            <button class="btn btn-link" type="button" data-bs-toggle="collapse" data-bs-target="#advancedOptions">
                Advanced Processing Options
            </button>
        </h5>
    </div>
    <div id="advancedOptions" class="collapse">
        <div class="card-body">
            <!-- Advanced options go here -->
        </div>
    </div>
</div>
```

### 3. Enhanced Frame Interval Selector

**Current**:
```html
<input type="number" class="form-control" id="frameInterval" name="frameInterval" value="1" min="1" max="60">
```

**Improved**:
```html
<div class="mb-3">
    <label for="frameInterval" class="form-label">Frame Extraction Interval</label>
    <input type="number" class="form-control" id="frameInterval" name="frameInterval" value="1.0" min="0.1" max="60" step="0.1">
    <div class="form-text">Extract one frame every X seconds (default: 1.0)</div>
</div>
```

### 4. Enhanced Confidence Threshold

**Current**:
- Default value: 50
- Range: 0-100

**Improved**:
```html
<div class="mb-3">
    <label for="confidenceThreshold" class="form-label">Confidence Threshold</label>
    <input type="range" class="form-range" id="confidenceThreshold" name="confidenceThreshold" min="0" max="100" value="75">
    <div class="form-text">Minimum confidence for matches: <span id="thresholdValue">75</span>%</div>
</div>
```

### 5. Model Selection

**Location**: In Advanced Processing Options

**Design**:
```html
<div class="mb-3">
    <label for="modelSelection" class="form-label">Model Selection</label>
    <select class="form-select" id="modelSelection" name="modelSelection">
        <option value="clip-base" selected>CLIP ViT-Base (faster, less accurate)</option>
        <option value="clip-large">CLIP ViT-Large (slower, more accurate)</option>
        <option value="siglip">SigLIP (experimental)</option>
    </select>
    <div class="form-text">Select the model for image comparison</div>
</div>
```

### 6. Processing Features Toggle

**Location**: In Advanced Processing Options

**Design**:
```html
<div class="mb-3">
    <label class="form-label">Processing Features</label>
    <div class="form-check">
        <input class="form-check-input" type="checkbox" id="useNormalization" name="useNormalization" checked>
        <label class="form-check-label" for="useNormalization">
            Apply image normalization (CLAHE)
        </label>
    </div>
    <div class="form-check">
        <input class="form-check-input" type="checkbox" id="useBatching" name="useBatching" checked>
        <label class="form-check-label" for="useBatching">
            Use batch processing
        </label>
    </div>
    <div class="form-check">
        <input class="form-check-input" type="checkbox" id="useClustering" name="useClustering" checked>
        <label class="form-check-label" for="useClustering">
            Apply temporal clustering
        </label>
    </div>
    <div class="form-check">
        <input class="form-check-input" type="checkbox" id="useAdaptiveThreshold" name="useAdaptiveThreshold">
        <label class="form-check-label" for="useAdaptiveThreshold">
            Use adaptive thresholding
        </label>
    </div>
</div>
```

### 7. Real-Time Progress Viewer

**Location**: Below the analyze button, visible during processing

**Design**:
```html
<div id="progressContainer" class="mt-3" style="display: none;">
    <div class="card">
        <div class="card-header">
            <h5>Analysis Progress</h5>
        </div>
        <div class="card-body">
            <div class="mb-2">
                <strong>Current Video:</strong> <span id="currentVideo">-</span>
            </div>
            <div class="mb-2">
                <strong>Progress:</strong> <span id="progressPercent">0</span>%
            </div>
            <div class="progress">
                <div id="progressBar" class="progress-bar" role="progressbar" style="width: 0%"></div>
            </div>
            <div class="mt-2">
                <strong>Matches Found:</strong> <span id="matchesFound">0</span>
            </div>
        </div>
    </div>
</div>
```

### 8. Results Display Enhancements

**Current Issues**:
- No distinction between positive and negative matches
- No clustering information
- No confidence score context

**Improvements**:
```html
<!-- In results.html -->
<table class="table table-striped">
    <thead>
        <tr>
            <th>Timestamp</th>
            <th>Confidence Score</th>
            <th>Delta Score</th>
            <th>Reference Image</th>
            <th>Cluster ID</th>
        </tr>
    </thead>
    <tbody>
        {% for match in matches %}
            <tr class="{{ 'table-success' if match.is_positive else 'table-warning' }}">
                <td class="timestamp">{{ "%.2f"|format(match.timestamp) }}s</td>
                <td class="confidence">{{ "%.2f"|format(match.confidence * 100) }}%</td>
                <td class="delta">{{ "%.2f"|format(match.delta_score) if match.delta_score else 'N/A' }}</td>
                <td>
                    {% if match.reference_image %}
                        <span class="badge bg-secondary">{{ match.reference_image|basename }}</span>
                    {% endif %}
                </td>
                <td>{{ match.cluster_id if match.cluster_id else 'N/A' }}</td>
            </tr>
        {% endfor %}
    </tbody>
</table>
```

## JavaScript Updates Required

### 1. Preview for Negative References

```javascript
function previewNegativeReferences() {
    const negativeInput = document.getElementById('negative_references');
    const preview = document.getElementById('negativePreview');
    
    if (negativeInput && preview) {
        preview.innerHTML = '';
        
        for (let i = 0; i < negativeInput.files.length; i++) {
            const file = negativeInput.files[i];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.className = 'thumbnail-preview img-thumbnail me-2 mb-2';
                    img.alt = file.name;
                    preview.appendChild(img);
                };
                reader.readAsDataURL(file);
            }
        }
    }
}
```

### 2. Real-Time Progress Updates

```javascript
function setupProgressTracking() {
    // This would connect to WebSocket or Server-Sent Events
    // For now, we'll simulate progress updates
    const progressContainer = document.getElementById('progressContainer');
    if (progressContainer) {
        progressContainer.style.display = 'block';
    }
}

// Update progress display
function updateProgress(videoName, percent, matches) {
    document.getElementById('currentVideo').textContent = videoName;
    document.getElementById('progressPercent').textContent = percent;
    document.getElementById('progressBar').style.width = percent + '%';
    document.getElementById('matchesFound').textContent = matches;
}
```

### 3. Enhanced Form Handling

```javascript
function handleFormSubmission() {
    const form = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (form && analyzeBtn) {
        form.addEventListener('submit', function() {
            analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            analyzeBtn.disabled = true;
            
            // Show progress container
            setupProgressTracking();
        });
    }
}
```

## CSS Updates Required

### 1. Thumbnail Preview Styles

```css
.thumbnail-preview {
    max-width: 100px;
    max-height: 100px;
    object-fit: cover;
}

#negativePreview .thumbnail-preview {
    border: 2px solid #dc3545; /* Red border for negative references */
}

#referencePreview .thumbnail-preview {
    border: 2px solid #28a745; /* Green border for positive references */
}
```

### 2. Progress Container Styles

```css
#progressContainer {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 300px;
    z-index: 1000;
}
```

## Responsive Design Considerations

1. **Mobile Layout**:
   - Collapse advanced options by default
   - Stack form elements vertically
   - Ensure touch-friendly controls

2. **Tablet Layout**:
   - Two-column layout for form elements
   - Keep advanced options collapsed but easily accessible

3. **Desktop Layout**:
   - Full-width form with advanced options in sidebar
   - Real-time progress in corner overlay

## Accessibility Improvements

1. **Screen Reader Support**:
   - Proper ARIA labels for all interactive elements
   - Status updates for processing progress
   - Clear error messaging

2. **Keyboard Navigation**:
   - Tab order optimization
   - Keyboard shortcuts for common actions
   - Focus indicators for all interactive elements

3. **Color Contrast**:
   - Ensure sufficient contrast for all text
   - Color-blind friendly color schemes
   - Alternative visual indicators

## User Experience Enhancements

1. **Tooltips and Help Text**:
   - Explanations for advanced features
   - Model selection guidance
   - Performance implications of settings

2. **Loading States**:
   - Visual feedback during file processing
   - Estimated time remaining
   - Cancel option for long processes

3. **Error Handling**:
   - Clear error messages for file upload issues
   - Guidance for resolving common problems
   - Graceful degradation when features aren't supported

## Implementation Priority

1. **Critical UI Changes**:
   - Negative reference upload
   - Decimal frame interval
   - 75% default confidence

2. **Important UI Changes**:
   - Real-time progress viewer
   - Enhanced results display
   - Advanced options section

3. **Nice-to-Have UI Changes**:
   - Model selection
   - Feature toggle switches
   - Enhanced styling

## Testing Plan

1. **Cross-Browser Testing**:
   - Chrome, Firefox, Safari, Edge
   - Mobile browsers (iOS Safari, Android Chrome)

2. **Responsive Testing**:
   - Various screen sizes
   - Mobile orientation changes
   - Touch vs. mouse interactions

3. **Accessibility Testing**:
   - Screen reader compatibility
   - Keyboard navigation
   - Color contrast verification

4. **User Testing**:
   - Usability study with new users
   - Feedback from current users
   - A/B testing for major changes