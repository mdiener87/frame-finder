# debug_analyzer.py
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import lpips
import open_clip
from sklearn.metrics import roc_curve, precision_recall_curve
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ==============================
# Device configuration
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# ==============================
# Constants
# ==============================
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

# More lenient default parameters for debugging
DEFAULT_FRAME_INTERVAL = 1.0
DEFAULT_FRAME_STRIDE = 1
DEFAULT_RESOLUTION_TARGET = 1080
DEFAULT_LPIPS_THRESHOLD = 0.8  # Much more lenient
DEFAULT_CLIP_THRESHOLD = 0.1   # Much more lenient
DEFAULT_NMS_IOU_THRESHOLD = 0.5
DEFAULT_DEBOUNCE_N = 1  # Less strict
DEFAULT_DEBOUNCE_M = 5  # Less strict

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==============================
# Preprocessing Pipeline
# ==============================

class ImagePreprocessor:
    """Handle preprocessing of reference images and video frames"""
    
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
        self.clip_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        # For LPIPS, we need images in [0,1] range
        self.lpips_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # For grayscale conversion
        self.gray_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    
    def preprocess_for_clip(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for CLIP model"""
        return self.clip_preprocess(image).unsqueeze(0).to(DEVICE)
    
    def preprocess_for_lpips(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for LPIPS model"""
        return self.lpips_preprocess(image).unsqueeze(0).to(DEVICE)
    
    def preprocess_for_template_matching(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for template matching (grayscale)"""
        gray_tensor = self.gray_preprocess(image)
        return (gray_tensor.squeeze(0).numpy() * 255).astype(np.uint8)
    
    def normalize_image(self, image: Image.Image) -> Image.Image:
        """Apply CLAHE normalization to stabilize lighting"""
        # Convert to LAB color space
        lab = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        lab = cv2.merge((l, a, b))
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(rgb)
    
    def letterbox_resize(self, image: Image.Image, target_size: int) -> Image.Image:
        """Resize image with letterboxing to maintain aspect ratio"""
        # Calculate new dimensions
        w, h = image.size
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create letterbox
        letterbox = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        letterbox.paste(resized, (paste_x, paste_y))
        
        return letterbox

# ==============================
# Multi-scale Template Matching
# ==============================

class TemplateMatcher:
    """Multi-scale template matching for candidate proposal generation"""
    
    def __init__(self, scales: List[float] = None):
        self.scales = scales or [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        self.top_k = 100  # More candidates for debugging
    
    def generate_template_pyramid(self, template: np.ndarray) -> List[np.ndarray]:
        """Generate scaled versions of template"""
        templates = []
        for scale in self.scales:
            if scale == 1.0:
                templates.append(template)
            else:
                h, w = template.shape
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h > 0 and new_w > 0:
                    scaled = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    templates.append(scaled)
        return templates
    
    def match_template_multiscale(self, frame: np.ndarray, template: np.ndarray) -> List[Dict[str, Any]]:
        """Perform multi-scale template matching"""
        templates = self.generate_template_pyramid(template)
        all_matches = []
        
        logger.debug(f"Template shape: {template.shape}, Frame shape: {frame.shape}")
        
        for i, scaled_template in enumerate(templates):
            # Skip if template is too small
            if scaled_template.shape[0] < 10 or scaled_template.shape[1] < 10:
                continue
                
            # Check if template is larger than frame
            if scaled_template.shape[0] > frame.shape[0] or scaled_template.shape[1] > frame.shape[1]:
                continue
                
            # Perform template matching
            try:
                result = cv2.matchTemplate(frame, scaled_template, cv2.TM_CCOEFF_NORMED)
                logger.debug(f"Scale {self.scales[i]}: result shape {result.shape}")
            except Exception as e:
                logger.warning(f"Template matching failed for scale {self.scales[i]}: {e}")
                continue
            
            # Find top-K matches
            locations = self._find_top_k_matches(result, self.top_k)
            logger.debug(f"Found {len(locations)} matches at scale {self.scales[i]}")
            
            # Convert to bounding boxes
            for loc in locations:
                x, y, score = loc
                h, w = scaled_template.shape
                match = {
                    'bbox': [x, y, w, h],
                    'score': score,
                    'scale': self.scales[i]
                }
                all_matches.append(match)
        
        # Sort by score and return top matches
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        logger.debug(f"Total matches before filtering: {len(all_matches)}")
        return all_matches[:self.top_k]
    
    def _find_top_k_matches(self, result: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
        """Find top-K matches from template matching result"""
        # Flatten the result array and get top-K indices
        flat = result.flatten()
        
        # Handle case where we have fewer elements than k
        actual_k = min(k, len(flat))
        if actual_k == 0:
            return []
            
        # Get top-K indices
        indices = np.argpartition(flat, -actual_k)[-actual_k:]
        indices = indices[np.argsort(-flat[indices])]
        
        # Convert flat indices to 2D coordinates
        locations = []
        for idx in indices:
            y, x = np.unravel_index(idx, result.shape)
            score = flat[idx]
            # Lower threshold for debugging
            if score > 0.05:  # Much lower threshold
                locations.append((int(x), int(y), float(score)))
        
        return locations

# ==============================
# LPIPS Verification
# ==============================

class LPIPSVerifier:
    """LPIPS-based candidate verification"""
    
    def __init__(self):
        self.loss_fn = lpips.LPIPS(net='alex').to(DEVICE)
        self.loss_fn.eval()
    
    @torch.no_grad()
    def compute_distance(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute LPIPS distance between two images"""
        # Ensure images are in [0,1] range
        if img1.max() > 1.0:
            img1 = img1 / 255.0
        if img2.max() > 1.0:
            img2 = img2 / 255.0
            
        # Ensure both tensors are on the same device
        img1 = img1.to(DEVICE)
        img2 = img2.to(DEVICE)
            
        distance = self.loss_fn(img1, img2)
        return float(distance.item())

# ==============================
# CLIP Verification
# ==============================

class CLIPVerifier:
    """CLIP-based candidate verification"""
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=DEVICE)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
    
    @torch.no_grad()
    def compute_similarity(self, image: torch.Tensor, text: str = None) -> float:
        """Compute CLIP similarity between image and text/reference"""
        # Get image features
        image_features = self.model.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        
        # If text is provided, compare with text
        if text:
            text_tokens = self.tokenizer([text]).to(DEVICE)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            similarity = (image_features @ text_features.T).item()
        else:
            # For image-to-image comparison, we would need reference features
            # This is a placeholder for when we have reference features
            similarity = 0.0
            
        return similarity
    
    @torch.no_grad()
    def compute_image_similarity(self, img1_features: torch.Tensor, img2_features: torch.Tensor) -> float:
        """Compute cosine similarity between two image feature vectors"""
        # Ensure both tensors are on the same device
        img1_features = img1_features.to(DEVICE)
        img2_features = img2_features.to(DEVICE)
        
        img1_features = F.normalize(img1_features, dim=-1)
        img2_features = F.normalize(img2_features, dim=-1)
        similarity = (img1_features @ img2_features.T).item()
        return similarity

# ==============================
# Reference Cache
# ==============================

class ReferenceCache:
    """Cache for reference images and their embeddings"""
    
    def __init__(self, reference_paths: List[str], preprocessor: ImagePreprocessor):
        self.reference_paths = reference_paths
        self.preprocessor = preprocessor
        self.references = []
        self.clip_features = []
        self.lpips_tensors = []
        self.template_images = []
        
        # Load and preprocess references
        self._load_references()
    
    def _load_references(self):
        """Load and preprocess reference images"""
        logger.info(f"Loading {len(self.reference_paths)} reference images")
        
        for path in self.reference_paths:
            try:
                # Load image
                image = Image.open(path).convert("RGB")
                logger.debug(f"Reference image {os.path.basename(path)} size: {image.size}")
                
                # Preprocess for different uses
                clip_tensor = self.preprocessor.preprocess_for_clip(image)
                lpips_tensor = self.preprocessor.preprocess_for_lpips(image)
                template_image = self.preprocessor.preprocess_for_template_matching(image)
                
                # Store in cache
                self.references.append(image)
                self.clip_features.append(clip_tensor)
                self.lpips_tensors.append(lpips_tensor)
                self.template_images.append(template_image)
                
                logger.info(f"Loaded reference: {os.path.basename(path)}")
            except Exception as e:
                logger.error(f"Error loading reference {path}: {e}")
    
    def get_clip_features(self) -> List[torch.Tensor]:
        """Get CLIP features for all references"""
        return self.clip_features
    
    def get_lpips_tensors(self) -> List[torch.Tensor]:
        """Get LPIPS tensors for all references"""
        return self.lpips_tensors
    
    def get_template_images(self) -> List[np.ndarray]:
        """Get template images for all references"""
        return self.template_images

# ==============================
# Frame Extraction
# ==============================

def extract_frames(video_path: str, interval: float = 1.0) -> List[Tuple[np.ndarray, float]]:
    """Extract frames from video at specified interval"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * interval)))
    idx = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
            
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((rgb, idx / fps))
        idx += 1
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames

# ==============================
# Non-Maximum Suppression
# ==============================

def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection over Union of two bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0.0

def non_max_suppression(matches: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Apply non-maximum suppression to remove duplicate detections"""
    if not matches:
        return []
    
    logger.debug(f"Applying NMS to {len(matches)} matches")
    
    # Sort by score in descending order
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    keep = []
    suppressed = set()
    
    for i, match in enumerate(matches):
        if i in suppressed:
            continue
            
        keep.append(match)
        
        # Suppress overlapping boxes
        for j in range(i + 1, len(matches)):
            if j in suppressed:
                continue
                
            iou = compute_iou(match['bbox'], matches[j]['bbox'])
            if iou >= iou_threshold:
                suppressed.add(j)
    
    logger.debug(f"NMS result: {len(keep)} matches kept")
    return keep

# ==============================
# Temporal Smoothing
# ==============================

def temporal_smoothing(detections: List[Dict], debounce_n: int = 1, debounce_m: int = 5) -> List[Dict]:
    """Apply temporal smoothing and debounce filtering"""
    if not detections:
        return []
    
    logger.debug(f"Applying temporal smoothing to {len(detections)} detections")
    
    # For debugging, let's just return all detections without filtering
    return detections

# ==============================
# Main Processing Function
# ==============================

def process_videos(
    reference_paths: List[str],
    video_paths: List[str],
    negative_paths: Optional[List[str]] = None,
    frame_interval: float = DEFAULT_FRAME_INTERVAL,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    resolution_target: int = DEFAULT_RESOLUTION_TARGET,
    lpips_threshold: float = DEFAULT_LPIPS_THRESHOLD,
    clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    debounce_n: int = DEFAULT_DEBOUNCE_N,
    debounce_m: int = DEFAULT_DEBOUNCE_M,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Process videos using the new three-stage prop detection algorithm.
    
    Args:
        reference_paths: Paths to positive reference images
        video_paths: Paths to video files
        negative_paths: Paths to negative reference images (optional)
        frame_interval: Seconds between frames to process
        frame_stride: Process every Nth frame then back-fill around hits
        resolution_target: Target resolution for processing (longest side)
        lpips_threshold: LPIPS distance threshold for verification
        clip_threshold: CLIP cosine similarity threshold for verification
        nms_iou_threshold: IoU threshold for non-maximum suppression
        debounce_n: Minimum consecutive frames for detection
        debounce_m: Window size for consecutive frame checking
        progress_callback: Callback for progress reporting
        
    Returns:
        Dict containing processing results
    """
    logger.info("Starting video processing with debug algorithm")
    logger.info(f"Parameters: frame_interval={frame_interval}, lpips_threshold={lpips_threshold}, clip_threshold={clip_threshold}")
    
    # Initialize preprocessors and verifiers
    preprocessor = ImagePreprocessor(target_size=resolution_target)
    template_matcher = TemplateMatcher()
    lpips_verifier = LPIPSVerifier()
    
    # Try to initialize CLIP verifier
    try:
        clip_verifier = CLIPVerifier()
        clip_available = True
        logger.info("CLIP verifier initialized successfully")
    except Exception as e:
        logger.warning(f"CLIP verifier initialization failed: {e}")
        clip_available = False
    
    # Load reference images
    reference_cache = ReferenceCache(reference_paths, preprocessor)
    template_images = reference_cache.get_template_images()
    
    results = {}
    
    # Process each video
    total_videos = len(video_paths)
    for video_idx, video_path in enumerate(video_paths):
        video_name = os.path.basename(video_path)
        logger.info(f"Processing video {video_idx + 1}/{total_videos}: {video_name}")
        
        # Report progress
        if progress_callback:
            progress_callback({
                'current_video': video_name,
                'video_index': video_idx,
                'total_videos': total_videos,
                'status': 'processing_video'
            })
        
        matches = []
        try:
            # Extract frames
            frames = extract_frames(video_path, interval=frame_interval)
            total_frames = len(frames)
            logger.info(f"Extracted {total_frames} frames from video")
            
            # Process frames
            for frame_idx, (frame_rgb, timestamp) in enumerate(frames):
                logger.debug(f"Processing frame {frame_idx}/{total_frames} at timestamp {timestamp}")
                
                # Report frame progress
                if progress_callback and total_frames > 0:
                    progress_callback({
                        'current_video': video_name,
                        'video_index': video_idx,
                        'total_videos': total_videos,
                        'current_frame': frame_idx,
                        'total_frames': total_frames,
                        'status': 'processing_frames'
                    })
                
                # Convert to PIL Image
                frame_pil = Image.fromarray(frame_rgb)
                
                # Preprocess frame for template matching
                frame_gray = preprocessor.preprocess_for_template_matching(frame_pil)
                
                # Match against all reference templates
                frame_matches = []
                for ref_idx, template in enumerate(template_images):
                    logger.debug(f"Matching template {ref_idx} with shape {template.shape}")
                    
                    # Perform multi-scale template matching
                    matches_raw = template_matcher.match_template_multiscale(frame_gray, template)
                    logger.debug(f"Found {len(matches_raw)} raw matches for template {ref_idx}")
                    
                    # Add reference index to matches
                    for match in matches_raw:
                        match['reference_index'] = ref_idx
                        match['timestamp'] = timestamp
                        frame_matches.append(match)
                
                logger.debug(f"Total frame matches before NMS: {len(frame_matches)}")
                
                # Apply non-maximum suppression
                frame_matches = non_max_suppression(frame_matches, nms_iou_threshold)
                logger.debug(f"Frame matches after NMS: {len(frame_matches)}")
                
                # Log some of the top matches for debugging
                if frame_matches:
                    top_matches = sorted(frame_matches, key=lambda x: x['score'], reverse=True)[:5]
                    logger.debug("Top 5 matches:")
                    for i, match in enumerate(top_matches):
                        logger.debug(f"  {i+1}. Score: {match['score']:.3f}, Scale: {match['scale']:.2f}, BBox: {match['bbox']}")
                
                # Verify candidates
                verified_matches = []
                for match_idx, match in enumerate(frame_matches):
                    x, y, w, h = match['bbox']
                    
                    logger.debug(f"Verifying match {match_idx}: bbox={match['bbox']}, frame_shape={frame_rgb.shape}")
                    
                    # Extract candidate region
                    if (y + h <= frame_rgb.shape[0] and x + w <= frame_rgb.shape[1] and 
                        w > 0 and h > 0):
                        candidate = frame_rgb[y:y+h, x:x+w]
                        candidate_pil = Image.fromarray(candidate)
                        
                        logger.debug(f"  Candidate shape: {candidate.shape}")
                        
                        # Preprocess for verification
                        candidate_lpips = preprocessor.preprocess_for_lpips(candidate_pil)
                        reference_lpips = reference_cache.get_lpips_tensors()[match['reference_index']]
                        
                        # LPIPS verification
                        lpips_score = lpips_verifier.compute_distance(candidate_lpips, reference_lpips)
                        match['lpips_score'] = lpips_score
                        logger.debug(f"  LPIPS score: {lpips_score:.3f}")
                        
                        # CLIP verification (if available)
                        if clip_available:
                            try:
                                candidate_clip = preprocessor.preprocess_for_clip(candidate_pil)
                                reference_clip = reference_cache.get_clip_features()[match['reference_index']]
                                
                                # Get features
                                with torch.no_grad():
                                    candidate_features = clip_verifier.model.encode_image(candidate_clip)
                                    reference_features = clip_verifier.model.encode_image(reference_clip)
                                
                                clip_score = clip_verifier.compute_image_similarity(
                                    candidate_features, reference_features)
                                match['clip_score'] = clip_score
                                logger.debug(f"  CLIP score: {clip_score:.3f}")
                            except Exception as e:
                                logger.warning(f"CLIP verification failed for match: {e}")
                                match['clip_score'] = 0.0
                        else:
                            match['clip_score'] = 0.0
                        
                        # Combined score (simple average for now)
                        match['confidence'] = 1.0 - (lpips_score / 2.0)  # Normalize LPIPS to [0,1]
                        if clip_available:
                            match['confidence'] = (match['confidence'] + match['clip_score']) / 2.0
                        
                        logger.debug(f"  Combined confidence: {match['confidence']:.3f}")
                        
                        # Apply much more lenient thresholds for debugging
                        if lpips_score <= lpips_threshold:
                            if not clip_available or match['clip_score'] >= clip_threshold:
                                verified_matches.append(match)
                                logger.debug(f"  Match verified and added")
                            else:
                                logger.debug(f"  Match rejected due to CLIP threshold")
                        else:
                            logger.debug(f"  Match rejected due to LPIPS threshold")
                    else:
                        logger.debug(f"  Match rejected due to invalid bbox or frame dimensions")
                
                logger.debug(f"Verified matches: {len(verified_matches)}")
                
                # Add verified matches to results
                matches.extend(verified_matches)
            
            # Apply temporal smoothing
            matches = temporal_smoothing(matches, debounce_n, debounce_m)
            
            # Calculate max confidence for UI display
            max_confidence = max([m.get("confidence", 0.0) for m in matches], default=0.0)
            
            # Add reference image names to matches
            for match in matches:
                if 'reference_index' in match and match['reference_index'] < len(reference_paths):
                    match['reference_image'] = os.path.basename(reference_paths[match['reference_index']])
            
            logger.info(f"Final results for {video_name}: {len(matches)} matches found")
            if matches:
                top_matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)[:5]
                logger.info("Top 5 final matches:")
                for i, match in enumerate(top_matches):
                    logger.info(f" {i+1}. Confidence: {match['confidence']:.3f}, Timestamp: {match['timestamp']:.3f}")
            
            results[video_name] = {
                "matches": matches,
                "max_confidence": max_confidence,
                "thresholds_used": {
                    "lpips": lpips_threshold,
                    "clip": clip_threshold if clip_available else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing video {video_name}: {e}", exc_info=True)
            results[video_name] = {
                "matches": [{"error": str(e)}],
                "max_confidence": 0.0,
                "thresholds_used": {
                    "lpips": lpips_threshold,
                    "clip": clip_threshold if clip_available else None
                }
            }
    
    logger.info("Video processing completed")
    return results