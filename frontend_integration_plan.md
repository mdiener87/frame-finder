# Frontend Integration Plan for New Prop Detection Algorithm

## Overview

This document outlines the changes needed to integrate the new prop detection algorithm with the existing frontend. While we're allowed to make breaking changes, we should maintain a similar user experience while enhancing functionality where beneficial.

## Current Frontend Analysis

### Existing Pages
1. **Index Page (`index.html`)** - File upload and configuration
2. **Results Page (`results.html`)** - Display processing results
3. **Base Template (`base.html`)** - Common layout and styling

### Current Features
- Reference image upload with preview
- Negative reference image upload with preview
- Video file upload (individual or directory)
- Frame interval configuration
- Progress tracking during processing
- Results display with timestamps and confidence scores
- Results export functionality
- Task cancellation

### Current UI Components
- File upload forms with previews
- Progress bar and status display
- Results tables with filtering
- Export functionality

## Proposed Frontend Enhancements

### New Configuration Options
We need to add UI elements for the new algorithm parameters:

#### Advanced Settings Panel
Add a collapsible advanced settings panel with:
1. **Frame Processing**
   - Frame stride (process every Nth frame)
   - Resolution target (longest side in pixels)

2. **Verification Thresholds**
   - LPIPS threshold slider (0.1-0.8, default 0.35)
   - CLIP threshold slider (0.1-0.8, default 0.33)

3. **Temporal Parameters**
   - NMS IoU threshold (0.1-0.9, default 0.5)
   - Debounce N (1-10, default 3)
   - Debounce M (5-30, default 12)

#### Presets
Add preset buttons for common use cases:
- High Precision (stricter thresholds)
- High Recall (looser thresholds)
- Balanced (default values)
- Performance (faster processing with trade-offs)

### Enhanced Results Display

#### Detailed Results Table
Enhance the results table to show:
1. Timestamp (existing)
2. Overall confidence score (existing)
3. LPIPS score (new)
4. CLIP score (new)
5. Bounding box coordinates (new)
6. Reference image (existing)

#### Visual Enhancements
1. Color coding based on individual scores
2. Bounding box overlay on thumbnails (if we generate them)
3. Processing statistics display
4. Confidence filtering by individual scores

#### New Visualization Options
1. Timeline view of detections
2. Confidence distribution charts
3. Processing statistics dashboard

## Implementation Plan

### Phase 1: Basic Integration
1. Update index.html to include new configuration options
2. Modify form submission to include new parameters
3. Maintain existing user workflow
4. Add advanced settings panel (collapsed by default)

### Phase 2: Enhanced Results Display
1. Update results.html to show additional information
2. Add filtering by individual scores
3. Implement visual enhancements
4. Add processing statistics display

### Phase 3: Advanced Features
1. Implement preset buttons
2. Add visualization options
3. Enhance user guidance and tooltips

## Detailed Changes

### Index Page (`index.html`) Updates

#### New Advanced Settings Panel
```html
<div class="card mt-3">
    <div class="card-header" id="advancedSettingsHeading">
        <h5 class="mb-0">
            <button class="btn btn-link" type="button" data-bs-toggle="collapse" data-bs-target="#advancedSettingsPanel" aria-expanded="false" aria-controls="advancedSettingsPanel">
                Advanced Settings
            </button>
        </h5>
    </div>
    <div id="advancedSettingsPanel" class="collapse" aria-labelledby="advancedSettingsHeading">
        <div class="card-body">
            <!-- Frame Processing -->
            <div class="mb-3">
                <label for="frameStride" class="form-label">Frame Stride</label>
                <input type="number" class="form-control" id="frameStride" name="frameStride" value="1" min="1" max="10">
                <div class="form-text">Process every Nth frame (default: 1)</div>
            </div>
            
            <div class="mb-3">
                <label for="resolutionTarget" class="form-label">Resolution Target</label>
                <input type="number" class="form-control" id="resolutionTarget" name="resolutionTarget" value="1080" min="480" max="4320">
                <div class="form-text">Target resolution for processing (longest side in pixels, default: 1080)</div>
            </div>
            
            <!-- Verification Thresholds -->
            <div class="mb-3">
                <label for="lpipsThreshold" class="form-label">LPIPS Threshold: <span id="lpipsThresholdValue">0.35</span></label>
                <input type="range" class="form-range" id="lpipsThreshold" name="lpipsThreshold" min="0.1" max="0.8" step="0.01" value="0.35">
                <div class="form-text">Lower values are more strict (default: 0.35)</div>
            </div>
            
            <div class="mb-3">
                <label for="clipThreshold" class="form-label">CLIP Threshold: <span id="clipThresholdValue">0.33</span></label>
                <input type="range" class="form-range" id="clipThreshold" name="clipThreshold" min="0.1" max="0.8" step="0.01" value="0.33">
                <div class="form-text">Higher values are more strict (default: 0.33)</div>
            </div>
            
            <!-- Temporal Parameters -->
            <div class="mb-3">
                <label for="nmsThreshold" class="form-label">NMS IoU Threshold: <span id="nmsThresholdValue">0.5</span></label>
                <input type="range" class="form-range" id="nmsThreshold" name="nmsThreshold" min="0.1" max="0.9" step="0.05" value="0.5">
                <div class="form-text">IoU threshold for non-maximum suppression (default: 0.5)</div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="debounceN" class="form-label">Debounce N: <span id="debounceNValue">3</span></label>
                        <input type="range" class="form-range" id="debounceN" name="debounceN" min="1" max="10" value="3">
                        <div class="form-text">Minimum consecutive frames (default: 3)</div>
                    </div>
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="debounceM" class="form-label">Debounce M: <span id="debounceMValue">12</span></label>
                        <input type="range" class="form-range" id="debounceM" name="debounceM" min="5" max="30" value="12">
                        <div class="form-text">Window size for checking (default: 12)</div>
                    </div>
                </div>
            </div>
            
            <!-- Presets -->
            <div class="mb-3">
                <label class="form-label">Presets</label>
                <div>
                    <button type="button" class="btn btn-outline-primary btn-sm preset-btn" data-preset="precision">High Precision</button>
                    <button type="button" class="btn btn-outline-primary btn-sm preset-btn" data-preset="recall">High Recall</button>
                    <button type="button" class="btn btn-outline-primary btn-sm preset-btn" data-preset="balanced">Balanced</button>
                    <button type="button" class="btn btn-outline-primary btn-sm preset-btn" data-preset="performance">Performance</button>
                </div>
            </div>
        </div>
    </div>
</div>
```

#### JavaScript Updates for New Controls
```javascript
// Add event listeners for new sliders
document.getElementById('lpipsThreshold').addEventListener('input', function() {
    document.getElementById('lpipsThresholdValue').textContent = this.value;
});

document.getElementById('clipThreshold').addEventListener('input', function() {
    document.getElementById('clipThresholdValue').textContent = this.value;
});

document.getElementById('nmsThreshold').addEventListener('input', function() {
    document.getElementById('nmsThresholdValue').textContent = this.value;
});

document.getElementById('debounceN').addEventListener('input', function() {
    document.getElementById('debounceNValue').textContent = this.value;
});

document.getElementById('debounceM').addEventListener('input', function() {
    document.getElementById('debounceMValue').textContent = this.value;
});

// Add event listeners for preset buttons
document.querySelectorAll('.preset-btn').forEach(button => {
    button.addEventListener('click', function() {
        applyPreset(this.dataset.preset);
    });
});

// Preset application function
function applyPreset(preset) {
    switch(preset) {
        case 'precision':
            // Stricter thresholds for high precision
            document.getElementById('lpipsThreshold').value = 0.25;
            document.getElementById('clipThreshold').value = 0.38;
            document.getElementById('nmsThreshold').value = 0.5;
            document.getElementById('debounceN').value = 4;
            document.getElementById('debounceM').value = 12;
            break;
        case 'recall':
            // Looser thresholds for high recall
            document.getElementById('lpipsThreshold').value = 0.40;
            document.getElementById('clipThreshold').value = 0.28;
            document.getElementById('nmsThreshold').value = 0.5;
            document.getElementById('debounceN').value = 2;
            document.getElementById('debounceM').value = 10;
            break;
        case 'balanced':
            // Default balanced settings
            document.getElementById('lpipsThreshold').value = 0.35;
            document.getElementById('clipThreshold').value = 0.33;
            document.getElementById('nmsThreshold').value = 0.5;
            document.getElementById('debounceN').value = 3;
            document.getElementById('debounceM').value = 12;
            break;
        case 'performance':
            // Settings optimized for performance
            document.getElementById('frameStride').value = 3;
            document.getElementById('resolutionTarget').value = 720;
            document.getElementById('lpipsThreshold').value = 0.35;
            document.getElementById('clipThreshold').value = 0.33;
            document.getElementById('nmsThreshold').value = 0.5;
            document.getElementById('debounceN').value = 2;
            document.getElementById('debounceM').value = 8;
            break;
    }
    
    // Update displayed values
    document.getElementById('lpipsThresholdValue').textContent = document.getElementById('lpipsThreshold').value;
    document.getElementById('clipThresholdValue').textContent = document.getElementById('clipThreshold').value;
    document.getElementById('nmsThresholdValue').textContent = document.getElementById('nmsThreshold').value;
    document.getElementById('debounceNValue').textContent = document.getElementById('debounceN').value;
    document.getElementById('debounceMValue').textContent = document.getElementById('debounceM').value;
}

// Update form submission to include new parameters
function handleFormSubmission() {
    // ... existing code ...
    
    // Add new parameters to formData
    const frameStride = document.getElementById('frameStride');
    const resolutionTarget = document.getElementById('resolutionTarget');
    const lpipsThreshold = document.getElementById('lpipsThreshold');
    const clipThreshold = document.getElementById('clipThreshold');
    const nmsThreshold = document.getElementById('nmsThreshold');
    const debounceN = document.getElementById('debounceN');
    const debounceM = document.getElementById('debounceM');
    
    if (frameStride) formData.append('frameStride', frameStride.value);
    if (resolutionTarget) formData.append('resolutionTarget', resolutionTarget.value);
    if (lpipsThreshold) formData.append('lpipsThreshold', lpipsThreshold.value);
    if (clipThreshold) formData.append('clipThreshold', clipThreshold.value);
    if (nmsThreshold) formData.append('nmsThreshold', nmsThreshold.value);
    if (debounceN) formData.append('debounceN', debounceN.value);
    if (debounceM) formData.append('debounceM', debounceM.value);
    
    // ... existing code ...
}
```

### Results Page (`results.html`) Updates

#### Enhanced Results Table
```html
<table class="table table-striped results-table" id="resultsTable-{{ loop.index }}">
    <thead>
        <tr>
            <th>Timestamp</th>
            <th>Overall Confidence</th>
            <th>LPIPS Score</th>
            <th>CLIP Score</th>
            <th>Reference Image</th>
        </tr>
    </thead>
    <tbody>
        {% for match in video_data.matches %}
            {% if match.error %}
                <tr class="table-danger">
                    <td colspan="5">Error: {{ match.error }}</td>
                </tr>
            {% else %}
                <tr class="{{ 'table-success' if match.confidence > 0.8 else 'table-warning' if match.confidence > 0.6 else 'table-secondary' }}"
                    data-confidence="{{ match.confidence }}"
                    data-lpips="{{ match.lpips_score|default(0) }}"
                    data-clip="{{ match.clip_score|default(0) }}">
                    <td class="timestamp" data-timestamp="{{ match.timestamp }}">{{ "%.3f"|format(match.timestamp) }}s</td>
                    <td class="confidence">{{ "%.2f"|format(match.confidence * 100) }}%</td>
                    <td class="lpips-score">{{ "%.2f"|format(match.lpips_score * 100) if match.lpips_score else 'N/A' }}%</td>
                    <td class="clip-score">{{ "%.2f"|format(match.clip_score * 100) if match.clip_score else 'N/A' }}%</td>
                    <td>
                        {% if match.reference_image %}
                            <span class="badge bg-secondary">{{ match.reference_image|basename }}</span>
                        {% endif %}
                    </td>
                </tr>
            {% endif %}
        {% endfor %}
    </tbody>
</table>
```

#### Enhanced Filtering Options
```html
<!-- Enhanced confidence filter -->
<div class="card mb-4">
    <div class="card-body">
        <div class="row">
            <div class="col-md-12">
                <h5>Confidence Filters</h5>
            </div>
            <div class="col-md-4">
                <div class="mb-2">
                    <label for="overallThreshold" class="form-label">Overall Confidence:</label>
                    <input type="range" class="form-range" id="overallThreshold" min="0" max="100" value="75">
                    <div class="form-text">
                        Minimum: <span id="overallThresholdValue">75</span>%
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="mb-2">
                    <label for="lpipsFilter" class="form-label">LPIPS Score:</label>
                    <input type="range" class="form-range" id="lpipsFilter" min="0" max="100" value="0">
                    <div class="form-text">
                        Minimum: <span id="lpipsFilterValue">0</span>%
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="mb-2">
                    <label for="clipFilter" class="form-label">CLIP Score:</label>
                    <input type="range" class="form-range" id="clipFilter" min="0" max="100" value="0">
                    <div class="form-text">
                        Minimum: <span id="clipFilterValue">0</span>%
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
```

#### Processing Statistics
```html
<!-- Processing statistics -->
<div class="card mb-4">
    <div class="card-header">
        <h5>Processing Statistics</h5>
    </div>
    <div class="card-body">
        <div class="row">
            <div class="col-md-3">
                <div class="stat-card">
                    <h6>Total Frames Processed</h6>
                    <p class="stat-value">{{ video_data.processing_stats.total_frames_processed|default(0) }}</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <h6>Total Proposals</h6>
                    <p class="stat-value">{{ video_data.processing_stats.total_proposals|default(0) }}</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <h6>Verifications</h6>
                    <p class="stat-value">{{ video_data.processing_stats.total_verifications|default(0) }}</p>
                </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <h6>Final Detections</h6>
                    <p class="stat-value">{{ video_data.processing_stats.final_detections|default(0) }}</p>
                </div>
            </div>
        </div>
    </div>
</div>
```

#### Enhanced JavaScript for Results Page
```javascript
// Enhanced filtering by individual scores
document.addEventListener('DOMContentLoaded', function() {
    // ... existing code ...
    
    // Add event listeners for new filters
    const lpipsFilter = document.getElementById('lpipsFilter');
    const clipFilter = document.getElementById('clipFilter');
    
    if (lpipsFilter) {
        lpipsFilter.addEventListener('input', function() {
            document.getElementById('lpipsFilterValue').textContent = this.value;
            applyFilters();
        });
    }
    
    if (clipFilter) {
        clipFilter.addEventListener('input', function() {
            document.getElementById('clipFilterValue').textContent = this.value;
            applyFilters();
        });
    }
    
    function applyFilters() {
        const overallThreshold = document.getElementById('overallThreshold').value;
        const lpipsThreshold = document.getElementById('lpipsFilter').value;
        const clipThreshold = document.getElementById('clipFilter').value;
        
        document.querySelectorAll('.results-table').forEach(function(resultsTable) {
            const videoIndex = resultsTable.id.replace('resultsTable-', '');
            const visibleCount = document.getElementById('visibleCount-' + videoIndex);
            
            let visibleRowCount = 0;
            resultsTable.querySelectorAll('tbody tr').forEach(function(row) {
                const overallConfidence = parseFloat(row.getAttribute('data-confidence')) * 100;
                const lpipsScore = parseFloat(row.getAttribute('data-lpips')) * 100 || 0;
                const clipScore = parseFloat(row.getAttribute('data-clip')) * 100 || 0;
                
                // Check all thresholds
                if (overallConfidence >= overallThreshold && 
                    lpipsScore >= lpipsThreshold && 
                    clipScore >= clipThreshold) {
                    row.style.display = '';
                    visibleRowCount++;
                } else {
                    row.style.display = 'none';
                }
            });
            
            // Update visible count for this video
            if (visibleCount) {
                visibleCount.textContent = visibleRowCount;
            }
        });
    }
});
```

## Backward Compatibility

While we can make breaking changes, we should maintain the same basic workflow:

1. Users can still upload files the same way
2. Processing still happens in the background
3. Results are still displayed in a tabular format
4. Export functionality still works

## User Experience Improvements

### Progressive Disclosure
- Keep advanced settings hidden by default
- Show only essential controls initially
- Allow users to expand advanced options when needed

### Presets for Common Use Cases
- Provide one-click optimization for different scenarios
- Help users understand the impact of different settings
- Reduce the learning curve for new users

### Enhanced Feedback
- More detailed progress reporting
- Processing statistics and performance metrics
- Visual indicators for different types of results

## Implementation Steps

### Step 1: Basic UI Updates
1. Add advanced settings panel to index.html
2. Update form submission JavaScript
3. Add new parameters to app.py processing

### Step 2: Enhanced Results Display
1. Update results.html with new table columns
2. Add filtering by individual scores
3. Display processing statistics

### Step 3: Advanced Features
1. Implement preset buttons
2. Add visual enhancements
3. Improve user guidance

### Step 4: Testing and Refinement
1. Test with existing data sets
2. Validate UI/UX improvements
3. Ensure responsive design
4. Update documentation