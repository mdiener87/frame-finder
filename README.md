# Frame Finder

A lightweight Flask-based web tool that identifies specific visual props in video files by comparing frames against reference images using image embedding similarity.

## Overview

Frame Finder extracts frames from MP4 video files at regular intervals and compares them against reference images using CLIP (Contrastive Language-Image Pre-training) model to find visual matches. While the initial use case is identifying the "Think Tank AI" prop from Star Trek: Voyager, the tool is designed to work with arbitrary inputs.

## Features

- Upload reference images (JPG/PNG) and video files (MP4)
- Extract frames from videos at configurable intervals
- Compare frames against reference images using CLIP similarity
- Display results with timestamps, confidence scores, and thumbnails
- Simple web interface for uploading and viewing results
- Optional export of results data

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd frame-finder
   ```

2. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser to http://localhost:5000

## Usage

1. Navigate to the web interface
2. Upload one or more reference images (JPG/PNG)
3. Upload one or more video files (MP4) or a directory containing videos
4. Adjust processing parameters if needed:
   - Frame extraction interval (default: 1 frame/second)
   - Confidence threshold (default: 50%)
5. Click "Analyze" to start processing
6. View results with timestamps, confidence scores, and thumbnails
7. Optionally export results

## Technical Details

### Core Components

- **Flask**: Web framework for handling requests and rendering templates
- **OpenCV**: Video processing and frame extraction
- **PIL/Pillow**: Image processing utilities
- **Transformers (Hugging Face)**: CLIP model integration for image similarity
- **PyTorch**: Underlying framework for CLIP model

### Processing Workflow

1. User uploads reference images and video files
2. Videos are processed to extract frames at regular intervals (default: 1 frame/second)
3. Each frame is compared against reference images using CLIP embeddings
4. Matches above a confidence threshold are collected
5. Results are displayed with timestamps, confidence scores, and thumbnails

### Configuration

- Frame extraction interval: Configurable in UI (1-60 seconds)
- Confidence threshold: Adjustable slider (0-100%)
- Thumbnail size: Configurable in image processing functions

## Development

### Project Structure

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
└── README.md              # This file
```

### Adding New Features

The modular architecture allows for easy extension:

- Add new comparison algorithms in analyzer.py
- Extend UI features in templates/
- Add new routes in app.py

### Code Structure

- `app.py`: Main Flask application with routes
- `analyzer.py`: Core processing logic
- `templates/`: HTML templates using Jinja2
- `static/`: CSS, JavaScript, and other static assets

## Future Enhancements

- Multi-threaded processing for faster analysis
- SQLite database for storing and querying results
- UI progress bar during analysis
- Support for additional video formats
- Batch processing for multiple videos
- Advanced filtering and sorting of results

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenAI CLIP model for image similarity
- Hugging Face Transformers for easy model integration
- Flask for the web framework