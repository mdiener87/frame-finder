# Dependencies Analysis for New Prop Detection Algorithm

## Current Dependencies (from requirements.txt)
```
Flask==2.3.2
opencv-python==4.8.0.74
Pillow==10.0.0
numpy==1.24.3
torch==2.8.0
transformers==4.33.1
requests==2.31.0
```

## Required Additional Dependencies for New Implementation

### Core Libraries
1. **lpips** - Perceptual similarity metrics
   - Version: >=0.1.4
   - Purpose: LPIPS distance calculation for candidate verification

2. **open-clip-torch** - CLIP embeddings
   - Version: >=2.0.0
   - Purpose: CLIP model integration for semantic similarity

3. **scikit-learn** - Machine learning utilities
   - Version: >=1.0.0
   - Purpose: ROC/PR curves for threshold calibration

### Optional Libraries
4. **ffmpeg-python** - Video processing
   - Version: >=0.2.0
   - Purpose: More efficient frame extraction (optional improvement)

5. **kornia** - Computer vision utilities
   - Version: >=0.6.0
   - Purpose: Image normalization operations

6. **matplotlib** - Visualization
   - Version: >=3.5.0
   - Purpose: Debug plots and visualization

7. **scikit-image** - Image processing
   - Version: >=0.19.0
   - Purpose: SSIM and other image quality metrics

## Updated Requirements List

For the new implementation, we'll need to update requirements.txt to include:

```
Flask==2.3.2
opencv-python==4.8.0.74
Pillow==10.0.0
numpy==1.24.3
torch==2.8.0
transformers==4.33.1
requests==2.31.0
lpips==0.1.4
open-clip-torch==2.20.0
scikit-learn==1.3.0
```

The optional libraries can be added later if needed for specific features or performance improvements.