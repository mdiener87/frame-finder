# Frame Finder - Requirements

## Python Dependencies

### Core Dependencies

1. **Flask** - Web framework for the application
   - Version: >= 2.0.0
   - Purpose: Handle HTTP requests, routing, and template rendering

2. **OpenCV (cv2)** - Video processing library
   - Package: opencv-python
   - Version: >= 4.5.0
   - Purpose: Extract frames from video files

3. **Pillow (PIL)** - Image processing library
   - Package: Pillow
   - Version: >= 8.0.0
   - Purpose: Image manipulation and format conversion

4. **NumPy** - Numerical computing library
   - Package: numpy
   - Version: >= 1.19.0
   - Purpose: Array operations for image data

5. **Transformers** - Hugging Face library for pre-trained models
   - Package: transformers
   - Version: >= 4.10.0
   - Purpose: CLIP model integration

6. **Torch** - PyTorch deep learning framework
   - Package: torch
   - Version: >= 1.9.0
   - Purpose: Underlying framework for CLIP model

7. **Requests** - HTTP library
   - Package: requests
   - Version: >= 2.25.0
   - Purpose: Downloading models if needed

### Development Dependencies

1. **Flake8** - Code linting
   - Package: flake8
   - Version: >= 3.8.0
   - Purpose: Code quality checking

2. **Black** - Code formatting
   - Package: black
   - Version: >= 21.0.0
   - Purpose: Code formatting consistency

## System Requirements

### Operating System
- Linux, macOS, or Windows
- Python 3.7 or higher

### Hardware
- Minimum: 4GB RAM (8GB recommended)
- CPU with at least 2 cores
- GPU recommended but not required (CPU processing supported)

### Storage
- At least 2GB free space for:
  - Python environment and dependencies
  - Temporary file storage during processing
  - Model downloads (CLIP models ~1-2GB)

## External Dependencies

### CLIP Model
- Default: `openai/clip-vit-base-patch32`
- Size: ~1.5GB
- Downloaded automatically on first use
- Internet connection required for initial download

## Installation Requirements

### Python Version
- Python 3.7 or higher
- pip package manager

### Virtual Environment (Recommended)
- venv or conda for environment isolation

### Network Access
- Internet connection for:
  - Downloading CLIP model on first run
  - Installing Python packages

## Optional Dependencies

### For Enhanced Performance
1. **Torchvision** - For optimized image processing
   - Package: torchvision
   - Version: >= 0.10.0

2. **Torch Audio** - For audio processing (if extending functionality)
   - Package: torchaudio
   - Version: >= 0.9.0

### For Database Integration (Stretch Goal)
1. **SQLite3** - Built into Python standard library
   - For storing and querying results

### For Multi-threading (Stretch Goal)
1. **Concurrent.futures** - Built into Python standard library
   - For parallel processing of videos

## Browser Requirements

### Supported Browsers
- Chrome (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Edge (latest 2 versions)

### Features Required
- HTML5 File API for file uploads
- JavaScript enabled
- CSS3 support

## File Format Support

### Input Formats
1. **Reference Images**
   - JPEG (.jpg, .jpeg)
   - PNG (.png)

2. **Video Files**
   - MP4 (.mp4)
   - Other formats supported by OpenCV (optional)

### Output Formats
1. **Thumbnails**
   - JPEG format

2. **Results Export**
   - JSON format
   - CSV format (optional)

## Performance Requirements

### Processing Time
- Frame extraction: ~10-30 frames/second (varies by hardware)
- Image comparison: ~1-5 frames/second (varies by hardware)
- Total processing time depends on:
  - Video length and resolution
  - Frame extraction interval
  - Hardware specifications

### Memory Usage
- Base application: ~500MB
- Model loading: ~1.5GB
- Video processing: Variable based on video resolution

## Security Requirements

### File Validation
- MIME type checking for uploaded files
- File extension validation
- Size limitations for uploads

### Input Sanitization
- Form data validation
- Path traversal prevention
- XSS prevention in templates

## Scalability Considerations

### Current Limitations
- Single-threaded processing
- In-memory result storage
- No persistent job queue

### Potential Enhancements
- Multi-threaded processing
- Database-backed storage
- Message queue for job processing