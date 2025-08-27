# Frame Finder UX Improvements Implementation Plan

## 1. Time Display Format Improvement

### Current Implementation
- Time is displayed in seconds with 2 decimal places: `{{ "%.2f"|format(match.timestamp) }}s`
- Found in `templates/results.html` line 39

### Proposed Solution
- Convert seconds to hh:mm:ss format using a JavaScript function
- Format: HH:MM:SS (e.g., 01:23:45 for 1 hour, 23 minutes, 45 seconds)

### Implementation Steps
1. Create a JavaScript function to convert seconds to hh:mm:ss format
2. Update the results template to use this function
3. Apply formatting to all timestamp displays

## 2. Settings Persistence Improvement

### Current Implementation
- No persistence of settings between page visits
- All settings reset when returning to main page

### Proposed Solution
- Store settings in browser localStorage when form is submitted
- Restore settings when loading the main page

### Implementation Steps
1. Add JavaScript to capture form values on submission
2. Store values in localStorage
3. Add JavaScript to restore values when page loads
4. Handle edge cases (clearing storage, invalid values)

## 3. Confidence Threshold & Dynamic Filtering Improvement

### Current Implementation Analysis
- Threshold filtering happens in `analyzer.py` at line 298: `if s >= (vid_thr if vid_thr is not None else -1.0):`
- This is applied during processing, discarding results below threshold
- Results are permanently filtered out

### Proposed Solution
- Store all results during processing (remove threshold filtering)
- Add a dynamic filtering slider on the results page
- Allow users to adjust threshold after processing to show/hide results

### Implementation Steps

#### Backend Changes (analyzer.py)
1. Modify `process_videos` function to collect all results regardless of threshold
2. Pass the threshold value to results for reference
3. Add a "max_confidence" value per video for UI display

#### Frontend Changes (results.html, main.js)
1. Add a slider control to filter results dynamically
2. Implement JavaScript to show/hide results based on slider value
3. Update UI to show max confidence per file
4. Add real-time update of result count as slider moves

## Detailed Implementation Plan

### Task 1: Time Format Conversion
1. Create `formatTime(seconds)` JavaScript function
2. Update `templates/results.html` to use formatted time
3. Test with various time values

### Task 2: Settings Persistence
1. Add `saveSettings()` function to `static/js/main.js`
2. Add `restoreSettings()` function to `static/js/main.js`
3. Call these functions at appropriate times
4. Update `templates/index.html` if needed for initialization

### Task 3: Dynamic Filtering
1. Modify `analyzer.py` to store all results:
   - Remove threshold filtering in match collection loop
   - Add max confidence tracking per video
   - Pass threshold value to results for reference
2. Update `templates/results.html`:
   - Add slider control for confidence threshold
   - Add result count display
   - Add max confidence display per file
3. Add JavaScript in `static/js/main.js`:
   - Implement dynamic filtering functionality
   - Add event listeners for slider
   - Update result display in real-time

## Technical Considerations

### Performance
- Storing all results may increase memory usage
- Dynamic filtering should be optimized for large result sets
- Consider pagination for very large result sets

### User Experience
- Slider should have smooth real-time updates
- Result count should update as slider moves
- Visual feedback when filtering is applied
- Clear indication of max confidence per file

### Backward Compatibility
- Changes should not break existing functionality
- Default behavior should remain similar to current implementation
- Error handling for edge cases

## File Modifications Summary

1. `analyzer.py`:
   - Modify `process_videos` function to collect all results
   - Add max confidence tracking

2. `static/js/main.js`:
   - Add time formatting function
   - Add settings persistence functions
   - Add dynamic filtering functionality

3. `templates/results.html`:
   - Update timestamp display format
   - Add confidence threshold slider
   - Add max confidence display per file
   - Add result count display

4. `templates/index.html`:
   - Add settings restoration functionality (if needed)