# Frame Finder UX Improvements - Technical Specification

## Overview
This document details the technical implementation for three UX improvements:
1. Display time index in hh:mm:ss format instead of seconds
2. Persist previous run settings when returning to main page
3. Implement dynamic confidence threshold filtering on results page

## 1. Time Display Format Improvement

### Current State
- Timestamps displayed as seconds with 2 decimal places in `templates/results.html` line 39:
  ```html
  <td class="timestamp">{{ "%.2f"|format(match.timestamp) }}s</td>
  ```

### Implementation Plan
- Create JavaScript function to convert seconds to hh:mm:ss format
- Update results template to use formatted time
- Ensure consistent formatting across all timestamp displays

### Technical Details
- Function signature: `formatTime(seconds)`
- Format: `HH:MM:SS` (pad with leading zeros as needed)
- Example: 3661.5 seconds → "01:01.500"

## 2. Settings Persistence Improvement

### Current State
- No persistence of form settings between page visits
- All settings reset when navigating away and back

### Implementation Plan
- Store form values in browser localStorage on form submission
- Restore values from localStorage when main page loads
- Handle edge cases (invalid values, missing storage)

### Technical Details
- Storage key: `frameFinderSettings`
- Values to store:
  - frameInterval (number)
  - confidenceThreshold (number)
- Functions needed:
  - `saveSettings()` - called on form submission
  - `restoreSettings()` - called on page load

## 3. Dynamic Confidence Threshold Filtering

### Current State Analysis
- Threshold filtering occurs in `analyzer.py` line 298:
  ```python
  if s >= (vid_thr if vid_thr is not None else -1.0):
  ```
- Results below threshold are permanently discarded
- No post-processing filtering capability

### Implementation Plan
- Modify analyzer to store ALL results regardless of threshold
- Add max confidence tracking per video
- Add dynamic filtering slider to results page
- Implement real-time result filtering in JavaScript

### Backend Changes (analyzer.py)

#### Modify `process_videos` function:
1. Remove threshold filtering when collecting matches (line 298)
2. Store all matches in temporary list
3. Track max confidence per video
4. Pass threshold value to results for reference

#### Updated Logic:
```python
# Instead of filtering during collection, collect all:
for ts, s, ridx in zip(frame_ts, scores_np, ref_idx_np):
    matches.append({
        "timestamp": float(ts),
        "confidence": float(s),
        "reference_image": os.path.basename(reference_paths[ridx]) if len(reference_paths) else None
    })

# Then apply temporal clustering
matches = cluster_peaks(matches, window_s=cluster_window)

# Track max confidence for UI display
max_confidence = max([m["confidence"] for m in matches], default=0.0)
```

#### Updated Return Value:
```python
results[video_name] = {
    "matches": matches,
    "max_confidence": max_confidence,
    "threshold_used": vid_thr
}
```

### Frontend Changes

#### Results Template (templates/results.html):
1. Add confidence threshold slider:
   ```html
   <div class="mb-3">
       <label for="dynamicThreshold" class="form-label">Filter Results by Confidence</label>
       <input type="range" class="form-range" id="dynamicThreshold" min="0" max="100" value="75">
       <div class="form-text">Minimum confidence: <span id="dynamicThresholdValue">75</span>%</div>
   </div>
   ```

2. Add result count display:
   ```html
   <div id="resultStats" class="mb-3">
       <span id="visibleCount">0</span> of <span id="totalCount">0</span> results displayed
   </div>
   ```

3. Add max confidence display per file:
   ```html
   <p class="text-muted">
       {{ matches|length }} match(es) found | 
       Max confidence: <span class="max-confidence">{{ "%.2f"|format(max_confidence * 100) }}%</span>
   </p>
   ```

#### JavaScript (static/js/main.js):
1. Add dynamic filtering functionality:
   ```javascript
   function filterResults(threshold) {
       // Hide/show table rows based on confidence threshold
       // Update visible count display
   }
   
   function updateResultStats() {
       // Update visible/total count display
   }
   ```

2. Add event listeners for slider:
   ```javascript
   document.getElementById('dynamicThreshold').addEventListener('input', function() {
       const threshold = this.value;
       document.getElementById('dynamicThresholdValue').textContent = threshold;
       filterResults(threshold / 100.0);
   });
   ```

## File Modification Summary

### analyzer.py
- Modify `process_videos` function to collect all results
- Update return format to include max confidence and threshold used
- Remove threshold filtering during match collection

### static/js/main.js
- Add `formatTime(seconds)` function
- Add `saveSettings()` and `restoreSettings()` functions
- Add dynamic filtering functionality
- Add event listeners for new UI elements

### templates/results.html
- Update timestamp display to use formatted time
- Add confidence threshold slider
- Add result statistics display
- Add max confidence display per file
- Add necessary JavaScript initialization

### templates/index.html
- Add settings restoration functionality (if needed)

## Implementation Order
1. Time formatting (least complex)
2. Settings persistence (moderate complexity)
3. Dynamic filtering (most complex, affects both frontend and backend)

## Testing Considerations
- Verify time formatting with various input values (0, 60, 3600, 3661.5)
- Test settings persistence across page reloads
- Verify dynamic filtering works with various threshold values
- Ensure backward compatibility with existing functionality
- Test edge cases (no results, single result, many results)