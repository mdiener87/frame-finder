# Frame Finder UX Improvements - Technical Specification

## Overview
This document details the technical implementation for the following UX improvements:
1. Remove confidence interval slider from main page
2. Create collapsible panels for settings and progress
3. Implement panel state management during analysis
4. Disable settings during analysis
5. Add cancel button with confirmation to progress panel
6. Fix dynamic confidence filter to apply to all videos

## 1. Remove Confidence Interval Slider

### Current State
- Confidence interval slider exists in `templates/index.html` lines 48-52
- Slider updates a text display of the threshold value

### Implementation Plan
- Remove the slider element and associated text display
- Keep the underlying functionality in JavaScript for potential future use
- Update form submission to not send confidence threshold value

## 2. Collapsible Panels Implementation

### Current State
- All form elements are in a single card in `templates/index.html`
- Progress display is a separate div that is shown/hidden with JavaScript

### Implementation Plan
- Create two Bootstrap collapse panels:
  1. Settings panel containing all form elements
  2. Progress panel containing progress display
- Add panel headers with toggle buttons
- Implement state management for panel visibility

### Technical Details
#### HTML Structure
```html
<!-- Settings Panel -->
<div class="card">
  <div class="card-header" id="settingsHeading">
    <h5 class="mb-0">
      <button class="btn btn-link" type="button" data-bs-toggle="collapse" data-bs-target="#settingsPanel">
        Settings
      </button>
    </h5>
  </div>
  <div id="settingsPanel" class="collapse show" data-bs-parent="#accordion">
    <div class="card-body">
      <!-- All current form elements except confidence slider -->
    </div>
  </div>
</div>

<!-- Progress Panel -->
<div class="card">
  <div class="card-header" id="progressHeading">
    <h5 class="mb-0">
      <button class="btn btn-link" type="button" data-bs-toggle="collapse" data-bs-target="#progressPanel">
        Progress
      </button>
    </h5>
  </div>
  <div id="progressPanel" class="collapse" data-bs-parent="#accordion">
    <div class="card-body">
      <!-- Current progress display elements -->
    </div>
  </div>
</div>
```

## 3. Panel State Management During Analysis

### Current State
- Progress container is shown/hidden with inline styles
- No panel collapsing functionality exists

### Implementation Plan
- When analysis starts:
  - Collapse settings panel
  - Expand progress panel
- When analysis completes or is canceled:
  - Expand settings panel
  - Collapse progress panel

### Technical Details
- Modify `handleFormSubmission()` in `static/js/main.js`
- Add functions to control panel states:
  ```javascript
  function collapseSettingsPanel() { /* implementation */ }
  function expandProgressPanel() { /* implementation */ }
  function expandSettingsPanel() { /* implementation */ }
  function collapseProgressPanel() { /* implementation */ }
  ```

## 4. Disable Settings During Analysis

### Current State
- Form elements remain enabled during analysis
- Only the submit button is disabled

### Implementation Plan
- Disable all form inputs, selects, and buttons during analysis
- Re-enable all elements when analysis completes or is canceled

### Technical Details
- Add function to disable/enable form elements:
  ```javascript
  function disableSettings(disable) { 
    // Iterate through all form elements and set disabled property
  }
  ```

## 5. Cancel Button with Confirmation

### Current State
- No cancel functionality exists
- Analysis runs until completion or error

### Implementation Plan
- Add cancel button to progress panel
- Implement confirmation dialog before canceling
- Add backend support for canceling tasks

### Technical Details
#### Frontend
- Add cancel button to progress panel:
  ```html
  <button id="cancelBtn" class="btn btn-warning">Cancel</button>
  ```
- Add event listener for cancel button:
  ```javascript
  document.getElementById('cancelBtn').addEventListener('click', function() {
    // Show confirmation dialog
    // If confirmed, send cancel request to backend
  });
  ```

#### Backend
- Add task cancellation support in `app.py`:
  ```python
  # Add cancellation flag to task data
  processing_tasks[task_id] = {
    'status': 'processing',
    'progress': 0,
    'results': None,
    'cancelled': False,  # New field
    # ... other fields
  }
  
  # Add cancellation endpoint
  @app.route('/cancel_task/<task_id>', methods=['POST'])
  def cancel_task(task_id):
    # Set cancelled flag
    # Return success response
  ```
  
- Modify processing function to check cancellation flag periodically

## 6. Fix Dynamic Confidence Filter for All Videos

### Current State
- Each video has its own confidence slider in `templates/results.html`
- JavaScript only handles the first slider due to ID-based selection

### Implementation Plan
- Update JavaScript to handle multiple sliders by class instead of ID
- Ensure all sliders work independently

### Technical Details
#### Current Issue
```javascript
// This only affects one slider
const slider = document.getElementById('dynamicThreshold-1');
```

#### Solution
```javascript
// Handle all sliders by class
document.querySelectorAll('.dynamic-threshold').forEach(function(slider) {
  // Apply filtering logic to corresponding table
});
```

## File Modification Summary

### templates/index.html
- Remove confidence interval slider
- Implement collapsible panels structure
- Add panel toggle functionality

### static/js/main.js
- Remove confidence threshold update functionality
- Add panel state management functions
- Add settings disable/enable functionality
- Add cancel button event listener and confirmation dialog
- Update form submission handling
- Add task cancellation request functionality

### templates/results.html
- No HTML changes needed, but JavaScript needs updating

### static/js/main.js (results page)
- Update dynamic filtering to work with all videos
- Fix selector logic to handle multiple sliders

### app.py
- Add task cancellation endpoint
- Add cancellation flag to task data structure
- Modify processing function to check cancellation flag

### static/css/style.css
- Add any necessary styles for new UI elements

## Implementation Order
1. Remove confidence interval slider
2. Create collapsible panels structure
3. Implement panel state management
4. Add settings disable/enable functionality
5. Add cancel button and confirmation dialog
6. Implement backend cancellation support
7. Fix dynamic confidence filter for all videos

## Testing Considerations
- Verify panel collapsing/expanding works correctly
- Ensure all form elements are properly disabled during analysis
- Test cancel functionality with confirmation dialog
- Verify cancellation actually stops processing
- Confirm dynamic filtering works for all videos independently
- Test edge cases (canceling immediately, canceling after completion, etc.)