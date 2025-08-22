# Frame Finder - UI Design

## User Interface Components

### 1. Main Upload Page

#### Header
- Project title: "Frame Finder"
- Brief description of functionality
- Navigation links (if any)

#### Upload Section
- Reference Image Upload
  - File input for images (JPG/PNG)
  - Support for multiple file selection
  - Preview of selected images
  - Validation feedback

- Video Upload
  - File input for videos (MP4)
  - Support for directory upload or multiple file selection
  - List of selected videos with names and sizes
  - Validation feedback

- Configuration Options (Optional)
  - Frame extraction interval slider/input
  - Confidence threshold slider/input
  - Processing options (single/multi-threaded)

- Action Buttons
  - "Analyze" button to start processing
  - "Reset" button to clear selections

#### Status Section
- Processing status indicator
- Progress information (when implemented)
- Error messages display

### 2. Results Page

#### Header
- Results title
- Summary information
  - Number of videos processed
  - Total matches found
  - Processing time

#### Results Display
- For each video:
  - Video name as section header
  - List of matches:
    - Thumbnail of matching frame
    - Timestamp in video
    - Confidence score (percentage)
    - Reference image that matched

#### Controls
- "Export Results" button (JSON/CSV)
- "Process More Videos" button (return to upload page)
- "Download Thumbnails" option

### 3. Error Pages
- File upload errors
- Processing errors
- File not found errors

## UI Layout Mockups

### Main Upload Page Layout

```
+-----------------------------------------------------+
| Frame Finder                               [Nav]    |
+-----------------------------------------------------+
|                                                     |
|  Identify specific visual props in video files     |
|                                                     |
+-----------------------------------------------------+
| Reference Images                                    |
| [Choose Files] +----------------------------------+  |
|                | Thumbnail | Thumbnail | Thumbnail|  |
|                |    1      |    2      |    3     |  |
|                +----------------------------------+  |
|                                                     |
| Video Files                                         |
| [Choose Files]                                      |
| +-------------------------------------------------+ |
| | video1.mp4 | 120 MB | [x]                        | |
| | video2.mp4 | 95 MB  | [x]                        | |
| +-------------------------------------------------+ |
|                                                     |
| Configuration (Optional)                            |
| Frame Interval: [1] second(s)                      |
| Confidence Threshold: [80]%                         |
|                                                     |
|              [ Analyze Videos ] [ Reset ]          |
|                                                     |
+-----------------------------------------------------+
| Status: Ready to process                           |
+-----------------------------------------------------+
```

### Results Page Layout

```
+-----------------------------------------------------+
| Frame Finder - Results                    [Nav]    |
+-----------------------------------------------------+
|                                                     |
|  Processing Complete                                |
|  2 videos processed, 15 matches found               |
|                                                     |
+-----------------------------------------------------+
| video1.mp4                                          |
| +------+--------+-------------+------------------+  |
| |Frame |Time    |Confidence   |Reference Image   |  |
| +------+--------+-------------+------------------+  |
| |[img] |00:05:23| 92.4%       |[ref1]            |  |
| |[img] |00:12:45| 87.1%       |[ref2]            |  |
| +------+--------+-------------+------------------+  |
|                                                     |
| video2.mp4                                          |
| +------+--------+-------------+------------------+  |
| |Frame |Time    |Confidence   |Reference Image   |  |
| +------+--------+-------------+------------------+  |
| |[img] |00:08:12| 95.7%       |[ref1]            |  |
| |[img] |00:22:34| 88.3%       |[ref1]            |  |
| +------+--------+-------------+------------------+  |
|                                                     |
| [ Export Results ] [ Process More Videos ]         |
|                                                     |
+-----------------------------------------------------+
```

## User Experience Flow

### Happy Path
1. User navigates to main page
2. User selects reference images
3. User selects video files
4. User clicks "Analyze"
5. System processes videos (showing progress)
6. System displays results
7. User views/export results

### Error Path
1. User uploads invalid file type
2. System shows error message
3. User corrects selection
4. User continues with valid files

## Responsive Design Considerations

### Desktop (> 1024px)
- Multi-column layout
- Larger thumbnails
- Full feature set

### Tablet (768px - 1024px)
- Adjusted column layout
- Medium thumbnails
- Full feature set

### Mobile (< 768px)
- Single column layout
- Smaller thumbnails
- Simplified controls
- Touch-friendly elements

## Accessibility Features

### Visual
- Sufficient color contrast
- Clear typography hierarchy
- Focus indicators for keyboard navigation

### Technical
- Semantic HTML structure
- ARIA labels for interactive elements
- Keyboard navigation support
- Screen reader compatibility

## Styling Guidelines

### Color Palette
- Primary: Blue (#007BFF) for actions and links
- Secondary: Gray (#6C757D) for secondary actions
- Success: Green (#28A745) for positive feedback
- Warning: Yellow (#FFC107) for warnings
- Danger: Red (#DC3545) for errors
- Background: Light gray (#F8F9FA)
- Text: Dark gray (#212529)

### Typography
- Base font: System sans-serif stack
- Font sizes:
  - Headers: 1.5rem - 2rem
  - Body: 1rem
  - Small text: 0.875rem

### Spacing
- Base unit: 1rem
- Padding: 1rem for containers
- Margin: 0.5rem - 1rem between elements

### Components
- Buttons: Rounded corners (0.25rem)
- Cards: Subtle shadows
- Inputs: Consistent styling with focus states

## JavaScript Functionality

### File Upload Handling
- Preview generation for images
- File validation
- Drag and drop support

### Form Validation
- File type checking
- Size limitations
- Required field validation

### Results Interaction
- Thumbnail lightbox (optional)
- Sorting capabilities
- Filtering by confidence/timestamp

### Progress Tracking (Future)
- Progress bar updates
- Estimated time remaining
- Cancel processing option