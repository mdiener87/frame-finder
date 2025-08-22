# Frame Finder - Project Summary

## Project Overview

Frame Finder is a Flask-based web application designed to identify specific visual props in video files by comparing extracted frames against reference images using image embedding similarity. While initially inspired by identifying the "Think Tank AI" prop from Star Trek: Voyager, the tool is built to work with arbitrary inputs.

## Key Features

1. **Reference Image Management**: Upload and manage reference images (JPG/PNG)
2. **Video Processing**: Handle MP4 video files or directories containing videos
3. **Frame Extraction**: Extract frames at configurable intervals (default: 1 frame/second)
4. **Image Similarity**: Compare frames against references using CLIP model
5. **Results Display**: Show matches with timestamps, confidence scores, and thumbnails
6. **Web Interface**: User-friendly UI for uploading, processing, and viewing results
7. **Export Capability**: Optional export of results data

## Technical Architecture

### Core Components

1. **Flask Web Application** (`app.py`)
   - Main application entry point
   - HTTP request handling
   - Template rendering
   - File upload management

2. **Analysis Engine** (`analyzer.py`)
   - Video frame extraction using OpenCV
   - Image preprocessing
   - CLIP model integration for similarity comparison
   - Results processing and formatting

3. **Web UI** (`templates/` and `static/`)
   - HTML templates using Jinja2
   - CSS styling
   - JavaScript enhancements
   - Responsive design

### Data Flow

```
User Uploads → File Validation → Video Processing → Frame Extraction → 
Image Comparison → Results Generation → UI Display → Optional Export
```

### Technology Stack

- **Backend**: Python 3.7+, Flask
- **Video Processing**: OpenCV
- **Image Processing**: PIL/Pillow, NumPy
- **AI/ML**: Transformers (Hugging Face), PyTorch (CLIP model)
- **Frontend**: HTML5, CSS3, JavaScript
- **Storage**: File system (with optional SQLite extension)

## Implementation Approach

### Phase 1: MVP Development

1. **Project Setup**
   - Directory structure creation
   - Requirements definition
   - Basic Flask application

2. **Core Functionality**
   - File upload handling
   - Video frame extraction
   - Basic image comparison
   - Results display

3. **UI Development**
   - Upload interface
   - Results presentation
   - Basic styling

### Phase 2: Enhancement

1. **Improved Processing**
   - CLIP model integration
   - Confidence scoring
   - Performance optimization

2. **Advanced UI**
   - Enhanced styling
   - Progress indicators
   - Configuration options

### Phase 3: Advanced Features

1. **Database Integration**
   - SQLite for result storage
   - Query capabilities

2. **Performance Improvements**
   - Multi-threaded processing
   - Memory management

3. **Export Functionality**
   - JSON/CSV export
   - Thumbnail downloads

## Project Structure

```
frame-finder/
├── app.py                 # Flask app entry point
├── analyzer.py            # Core processing logic
├── requirements.txt       # Python dependencies
├── templates/             # HTML templates
│   ├── base.html          # Base template
│   ├── index.html         # Main upload page
│   └── results.html       # Results display
├── static/                # Static assets
│   ├── css/               # Stylesheets
│   ├── js/                # JavaScript files
│   └── thumbnails/        # Generated thumbnails
├── README.md              # Project documentation
├── technical_spec.md      # Technical specifications
├── implementation_plan.md  # Implementation approach
├── requirements.md        # Dependency requirements
├── ui_design.md           # UI/UX design
└── roadmap.md             # Development roadmap
```

## Development Roadmap

### Short-term Goals (Week 1-2)
- [ ] Complete basic Flask application structure
- [ ] Implement video frame extraction
- [ ] Create simple image comparison
- [ ] Develop basic UI with upload and results pages

### Medium-term Goals (Week 2-3)
- [ ] Integrate CLIP model for accurate similarity comparison
- [ ] Enhance UI with better styling and user experience
- [ ] Add configuration options for processing parameters
- [ ] Implement basic error handling and validation

### Long-term Goals (Week 3-4)
- [ ] Add database integration for result storage
- [ ] Implement multi-threaded processing
- [ ] Add export functionality
- [ ] Comprehensive testing and optimization

## Success Criteria

### Functional Requirements
- Users can upload reference images and video files
- System processes videos and extracts frames
- Frames are compared against reference images
- Results are displayed with timestamps and confidence scores
- UI is intuitive and responsive

### Performance Requirements
- Frame extraction: 10+ frames/second
- Image comparison: 1+ frames/second
- Memory usage: < 2GB during processing
- Processing time: < 2x video length for standard inputs

### Quality Requirements
- Code coverage: > 80%
- Error handling for all user inputs
- Clear documentation and comments
- Responsive UI across device sizes

## Risk Mitigation

### Technical Risks
- **CLIP model integration complexity**: Start with simpler models and add CLIP as an enhancement
- **Video processing performance**: Implement progress tracking and batch processing
- **Memory usage**: Process videos sequentially and implement memory cleanup

### Schedule Risks
- **Dependency on third-party libraries**: Identify fallback options and alternatives
- **Hardware limitations**: Develop with performance constraints and optimization in mind

## Next Steps

1. Create the basic project structure and Flask application
2. Implement video frame extraction functionality
3. Develop the file upload interface
4. Create basic image comparison logic
5. Design and implement the user interface
6. Test with sample data
7. Iterate and enhance based on results

This project provides a solid foundation for visual prop identification in videos while maintaining a modular architecture that allows for future enhancements and extensions.