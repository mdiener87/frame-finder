# Frame Finder - Technical Specification

## Project Overview
Frame Finder is a Flask-based web tool that identifies specific visual props in video files by comparing frames against reference images using image embedding similarity.

## Core Features
1. Upload reference images (JPG/PNG)
2. Upload video files (MP4) or directories containing videos
3. Extract frames at regular intervals from videos
4. Compare frames to reference images using CLIP or similar
5. Display results with timestamps, confidence scores, and thumbnails
6. Simple web UI for interaction
7. Optional export of results

## Technology Stack
- Backend: Python Flask
- Image/Video Processing: OpenCV, PIL/Pillow
- Image Similarity: CLIP (OpenAI's Contrastive Language-Image Pre-training)
- Frontend: HTML/CSS/JavaScript with Flask templates

## Project Structure
```
frame-finder/
├── app.py                 # Flask app entry point
├── analyzer.py            # Core image/video processing logic
├── requirements.txt       # Python dependencies
├── templates/             # HTML templates
│   ├── base.html          # Base template
│   ├── index.html         # Main upload page
│   └── results.html       # Results display
├── static/                # Static assets
│   ├── css/               # Stylesheets
│   ├── js/                # JavaScript files
│   └── thumbnails/        # Generated thumbnails
└── README.md              # Project documentation
```

## Core Components

### 1. Flask App (app.py)
- Main application entry point
- Routes for:
  - Home page (upload interface)
  - Upload handling
  - Analysis processing
  - Results display

### 2. Analyzer Module (analyzer.py)
- Video frame extraction
- Image preprocessing
- Similarity comparison using CLIP
- Results processing and formatting

### 3. Web UI
- Simple, clean interface
- Upload forms for reference images and videos
- Progress indication during processing
- Results display with thumbnails

## Implementation Plan

### Phase 1: Basic Structure
- Set up Flask app with basic routes
- Create directory structure
- Implement simple upload functionality

### Phase 2: Core Processing Logic
- Implement video frame extraction
- Integrate CLIP model for similarity comparison
- Process and return results

### Phase 3: UI Development
- Create templates for upload and results pages
- Add styling for a clean interface
- Implement thumbnail generation

### Phase 4: Enhancement
- Add progress tracking
- Implement optional SQLite storage
- Add export functionality

## Dependencies
- Flask: Web framework
- OpenCV: Video processing
- PIL/Pillow: Image processing
- transformers: CLIP model integration
- torch: PyTorch for CLIP