# Frame Finder

A lightweight Flask-based web tool that identifies specific visual props in video files by comparing frames against reference images using image embedding similarity.

## Overview

Frame Finder extracts frames from MP4 video files at regular intervals and compares them against reference images using CLIP (Contrastive Language-Image Pre-training) model to find visual matches. While the initial use case is identifying the "Think Tank AI" prop from Star Trek: Voyager, the tool is designed to work with arbitrary inputs.

## Features

- Upload reference images (JPG/PNG) and video files (MP4)
- Upload negative reference images to reduce false positives
- Extract frames from videos at configurable intervals (supports decimal values)
- Compare frames against reference images using CLIP similarity with delta scoring
- Apply image normalization (CLAHE on L channel) for consistent lighting
- GPU acceleration for faster processing
- Batch frame processing for improved throughput
- Temporal clustering to reduce duplicate detections
- Adaptive thresholding per video for better accuracy
- Display results with timestamps, confidence scores, and thumbnails
- Real-time progress viewer during analysis
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
3. Optionally upload negative reference images (images that should NOT be detected)
4. Upload one or more video files (MP4)
5. Adjust processing parameters if needed:
   - Frame extraction interval (default: 1.0 frame/second, supports decimal values)
   - Confidence threshold (default: 75%)
6. Click "Analyze" to start processing
7. View real-time progress during analysis
8. View results with timestamps, confidence scores, and thumbnails
9. Optionally export results

## Technical Details

### Core Components

- **Flask**: Web framework for handling requests and rendering templates
- **OpenCV**: Video processing and frame extraction
- **PIL/Pillow**: Image processing utilities
- **Transformers (Hugging Face)**: CLIP model integration for image similarity
- **PyTorch**: Underlying framework for CLIP model

### Processing Workflow

1. User uploads reference images, negative reference images (optional), and video files
2. Reference embeddings are computed once and cached for efficiency
3. Videos are processed to extract frames at regular intervals (supports decimal values)
4. Each frame is normalized using CLAHE on L channel for consistent lighting
5. Each frame is compared against reference images using CLIP embeddings with delta scoring
6. Matches above a confidence threshold are collected
7. Temporal clustering is applied to reduce duplicate detections
8. Results are displayed with timestamps, confidence scores, and thumbnails

### Configuration

- Frame extraction interval: Configurable in UI (0.1-60 seconds, supports decimal values)
- Confidence threshold: Adjustable slider (0-100%, default: 75%)
- Negative reference images: Optional upload to reduce false positives
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

- SQLite database for storing and querying results
- Support for additional video formats
- Advanced filtering and sorting of results
- Model selection (CLIP-ViT-Large, SigLIP)
- Two-stage filtering (OpenCV gate → CLIP re-check)
- Micro-tuning around peaks

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenAI CLIP model for image similarity
- Hugging Face Transformers for easy model integration
- Flask for the web framework