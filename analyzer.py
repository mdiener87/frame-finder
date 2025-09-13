"""
Robust reference-frame finder using OpenCV feature matching + homography.

This replaces the prior CLIP-based logic to work fully offline and provide
clear, high-precision matches with calibrated confidences.
"""

import os
from typing import List, Dict, Any, Tuple
import uuid

import cv2
import numpy as np
from PIL import Image

# ==============================
# Allowed extensions
# ==============================
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4"}

# ==============================
# Scanning Mode Presets
# ==============================
SCANNING_MODES = {
    "fast": {
        "name": "Fast Scan",
        "description": "Quick analysis for obvious matches. Best for larger objects or initial exploration.",
        "when_to_use": "Use when scanning many videos or looking for obvious, large objects",
        "orb_params": {
            "nfeatures": 2000,
            "scaleFactor": 1.2,
            "nlevels": 8,
            "edgeThreshold": 25,
            "patchSize": 25
        },
        "template_scales": [0.6, 0.8, 1.0, 1.2, 1.5],
        "enable_preprocessing": False,
        "min_matches_factor": 25  # Looser requirement: len(desc) // 25
    },
    "balanced": {
        "name": "Balanced Scan", 
        "description": "Good balance of speed and accuracy. Suitable for most use cases.",
        "when_to_use": "Default choice for medium-sized objects and general analysis",
        "orb_params": {
            "nfeatures": 3500,
            "scaleFactor": 1.15,
            "nlevels": 10,
            "edgeThreshold": 20,
            "patchSize": 20
        },
        "template_scales": [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0],
        "enable_preprocessing": True,
        "min_matches_factor": 22  # len(desc) // 22
    },
    "thorough": {
        "name": "Thorough Scan",
        "description": "Maximum detail and accuracy. Best for small objects and difficult matches.",
        "when_to_use": "Use for tiny props (like 57x159px), complex scenes, or when initial scans miss objects",
        "orb_params": {
            "nfeatures": 5000,
            "scaleFactor": 1.1,
            "nlevels": 12,
            "edgeThreshold": 15,
            "patchSize": 15
        },
        "template_scales": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0],
        "enable_preprocessing": True,
        "min_matches_factor": 20  # len(desc) // 20
    }
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ==============================
# Frame extraction
# ==============================
def extract_frames(video_path: str, interval: float = 1.0) -> List[Tuple[Image.Image, float]]:
    """Return list of (PIL.Image, timestamp_seconds) sampled by modulo frame step."""
    frames: List[Tuple[Image.Image, float]] = []
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
            frames.append((Image.fromarray(rgb), idx / fps))
        idx += 1
    cap.release()
    return frames


# ==============================
# Frame preprocessing for enhanced detection
# ==============================
def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Enhance frame for better small object detection."""
    # Sharpen for better edge detection
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(frame_bgr, -1, kernel)
    
    # Contrast enhancement using CLAHE
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ==============================
# Template matching for small objects
# ==============================
def template_match_score(ref_gray: np.ndarray, frame_gray: np.ndarray, scales: List[float] = None) -> Tuple[float, Dict[str, Any]]:
    """Multi-scale template matching with configurable scales."""
    if scales is None:
        scales = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]
    
    best_score = 0.0
    best_scale = 1.0
    best_location = (0, 0)
    h_ref, w_ref = ref_gray.shape
    h_frame, w_frame = frame_gray.shape
    
    for scale in scales:
        scaled_w = int(w_ref * scale)
        scaled_h = int(h_ref * scale)
        
        # Skip if scaled template is larger than frame
        if scaled_w > w_frame or scaled_h > h_frame:
            continue
        
        # Skip if template becomes too small to be meaningful
        if scaled_w < 10 or scaled_h < 10:
            continue
            
        scaled_ref = cv2.resize(ref_gray, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        
        # Use normalized cross correlation
        result = cv2.matchTemplate(frame_gray, scaled_ref, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_scale = scale
            best_location = max_loc
    
    meta = {
        "template_score": float(best_score),
        "best_scale": float(best_scale),
        "location": best_location,
        "method": "template_matching"
    }
    
    return float(best_score), meta


# ==============================
# Feature matching + homography scoring
# ==============================
def _cuda_available() -> bool:
    """Return True if OpenCV CUDA is available and a device is present.
    Can be forced off via env FRAME_FINDER_USE_CUDA=0, or forced on with =1.
    """
    force = os.getenv("FRAME_FINDER_USE_CUDA")
    if force is not None:
        return force not in ("0", "false", "False")
    try:
        # In some builds, cv2.cuda may not exist
        if not hasattr(cv2, "cuda"):
            return False
        return int(cv2.cuda.getCudaEnabledDeviceCount() or 0) > 0
    except Exception:
        return False


USE_CUDA = _cuda_available()


class RefFeatures:
    def __init__(self, path: str, max_size: int = 1024, scanning_mode: str = "thorough"):
        self.path = path
        self.scanning_mode = scanning_mode
        self.mode_config = SCANNING_MODES.get(scanning_mode, SCANNING_MODES["thorough"])
        
        self.image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.image_bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        h, w = self.image_bgr.shape[:2]
        scale = min(1.0, max_size / max(h, w))
        if scale < 0.999:
            self.image_bgr = cv2.resize(self.image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        self.gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)

        # Use ORB parameters from scanning mode configuration
        orb_params = self.mode_config["orb_params"]
        self.detector = cv2.ORB_create(**orb_params)
        self.kp, self.desc = self.detector.detectAndCompute(self.gray, None)


def compute_frame_features(frame_bgr: np.ndarray, detector) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Compute keypoints/desc for a frame. If CUDA is available, try GPU ORB.
    Fallback to CPU ORB on any failure. Returns (keypoints, descriptors, gray_cpu).
    """
    if USE_CUDA and hasattr(cv2, "cuda") and hasattr(cv2.cuda, "ORB_create"):
        try:
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(frame_bgr)
            gpu_gray = cv2.cuda.cvtColor(gpu_src, cv2.COLOR_BGR2GRAY)
            # Create CUDA ORB with optimized parameters for small objects
            orb_gpu = cv2.cuda_ORB_create(nfeatures=5000)
            # detectAndComputeAsync returns (keypoints (CPU list), descriptors (GpuMat))
            kp, desc_gpu = orb_gpu.detectAndComputeAsync(gpu_gray, None)
            desc = desc_gpu.download() if desc_gpu is not None else None
            # Also keep a CPU gray copy for downstream homography
            gray_cpu = gpu_gray.download()
            if kp is not None and desc is not None:
                return kp, desc, gray_cpu
        except Exception:
            # Fall back to CPU path below
            pass

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kp, desc = detector.detectAndCompute(gray, None)
    return kp, desc, gray


def match_and_score(ref: RefFeatures, frame_bgr: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """Return (confidence [0..1], meta) for one ref vs frame using ORB+homography."""
    # Reuse the same ORB params as ref
    kp_f, desc_f, gray_f = compute_frame_features(frame_bgr, ref.detector)
    if desc_f is None or ref.desc is None or len(desc_f) < 8 or len(ref.desc) < 8:
        return 0.0, {"reason": "no_descriptors"}

    # KNN match + ratio test (CPU matcher; robust and compatible)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn = bf.knnMatch(ref.desc, desc_f, k=2)
    good = []
    for m, n in matches_knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Use adaptive minimum matches from scanning mode configuration
    min_matches_factor = ref.mode_config["min_matches_factor"]
    min_matches = max(4, min(10, len(ref.desc) // min_matches_factor))
    if len(good) < min_matches:
        return 0.0, {"reason": "too_few_good_matches", "good": len(good), "min_required": min_matches}

    src_pts = np.float32([ref.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return 0.0, {"reason": "no_homography", "good": len(good)}

    inliers = int(mask.sum())
    inlier_ratio = inliers / max(1, len(good))

    # Confidence calibration:
    # - Encourage higher inlier counts and ratios.
    # - Soft-cap counts at 40 to map into [0,1].
    c_count = min(1.0, inliers / 40.0)
    c_ratio = min(1.0, inlier_ratio / 0.6)  # 0.6 ratio ~ strong geometric consistency
    confidence = 0.7 * c_count + 0.3 * c_ratio

    # Additional sanity: ensure projected polygon is plausible (non-degenerate)
    h, w = ref.gray.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H)
    area = cv2.contourArea(proj.reshape(-1, 2))
    if area <= 10.0:
        confidence *= 0.2

    meta = {
        "inliers": inliers,
        "matches": len(good),
        "inlier_ratio": inlier_ratio,
        "proj_area": float(area),
    }
    return float(confidence), meta


def hybrid_match_score(ref: RefFeatures, frame_bgr: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """Combine feature matching and template matching based on scanning mode configuration."""
    
    # Use preprocessing based on scanning mode configuration
    if ref.mode_config["enable_preprocessing"]:
        enhanced_frame = preprocess_frame(frame_bgr)
    else:
        enhanced_frame = frame_bgr.copy()
    
    # Try feature matching
    feat_conf, feat_meta = match_and_score(ref, enhanced_frame)
    
    # For low feature confidence, try template matching as fallback
    if feat_conf < 0.4:  # Lower threshold to give template matching a chance
        frame_gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        template_scales = ref.mode_config["template_scales"]
        template_conf, template_meta = template_match_score(ref.gray, frame_gray, template_scales)
        
        # Use template matching if it's significantly better
        if template_conf > feat_conf * 1.2:  # 20% boost required to switch methods
            # Normalize template matching score to be more comparable with feature matching
            normalized_template = min(1.0, template_conf * 1.3)  # Slight boost for template scores
            template_meta["normalized_score"] = normalized_template
            template_meta["original_feature_score"] = feat_conf
            template_meta["scanning_mode"] = ref.scanning_mode
            return normalized_template, template_meta
    
    # Use feature matching result, but boost slightly if we got a decent template score
    if feat_conf > 0.0:
        frame_gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        template_scales = ref.mode_config["template_scales"]
        template_conf, _ = template_match_score(ref.gray, frame_gray, template_scales)
        if template_conf > 0.3:  # Decent template match
            feat_conf = min(1.0, feat_conf + template_conf * 0.1)  # Small boost
            feat_meta["template_boost"] = template_conf * 0.1
    
    feat_meta["enhanced_preprocessing"] = ref.mode_config["enable_preprocessing"]
    feat_meta["scanning_mode"] = ref.scanning_mode
    return feat_conf, feat_meta


def cluster_peaks(matches: List[Dict[str, Any]], window_s: float = 1.0) -> List[Dict[str, Any]]:
    """Collapse nearby timestamps; keep the highest-confidence per window."""
    if not matches:
        return []
    matches.sort(key=lambda m: m["timestamp"])
    out: List[Dict[str, Any]] = []
    cur = [matches[0]]
    for m in matches[1:]:
        if m["timestamp"] - cur[-1]["timestamp"] <= window_s:
            cur.append(m)
        else:
            out.append(max(cur, key=lambda x: x["confidence"]))
            cur = [m]
    out.append(max(cur, key=lambda x: x["confidence"]))
    return out


# ==============================
# Public API: process_videos
# ==============================
def process_videos(reference_paths: List[str],
                   video_paths: List[str],
                   frame_interval: float = 1.0,
                   confidence_threshold: float = 0.75,
                   scanning_mode: str = "thorough",
                   progress_callback=None,
                   top_preview_count: int = 5) -> Dict[str, Any]:
    """
    Args:
        reference_paths: paths to positive reference images
        video_paths    : paths to mp4 files
        frame_interval : seconds between frames extracted
        confidence_threshold: retained for UI compatibility; analyzer returns all matches
        scanning_mode: scanning intensity mode ("fast", "balanced", "thorough")
        progress_callback: optional callable for UI progress updates
        top_preview_count: number of top matches to persist as preview images (0 disables)

    Returns:
        { video_name: { matches: [...], max_confidence: float, threshold_used: float, scanning_mode: str } }
    """
    # Precompute reference features once with the specified scanning mode
    refs: List[RefFeatures] = [RefFeatures(p, scanning_mode=scanning_mode) for p in (reference_paths or [])]

    results: Dict[str, Any] = {}
    # Ensure thumbnail output directory exists
    thumb_dir = os.path.join("static", "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)

    # Wipe previous run thumbnails to avoid accumulation
    try:
        for fn in os.listdir(thumb_dir):
            fp = os.path.join(thumb_dir, fn)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except Exception:
                    pass
    except Exception:
        pass
    total_videos = len(video_paths)

    for i, video_path in enumerate(video_paths):
        video_name = os.path.basename(video_path)
        matches: List[Dict[str, Any]] = []
        max_conf = 0.0
        try:
            if progress_callback:
                progress_callback({
                    "current_video": video_name,
                    "video_index": i,
                    "total_videos": total_videos,
                    "status": "processing_video",
                })

            frames = extract_frames(video_path, interval=max(0.1, float(frame_interval)))
            total_frames = len(frames)

            # Store only matched frames we may need to persist later
            matched_frames: Dict[int, np.ndarray] = {}

            for j, (pil_img, ts) in enumerate(frames):
                frame_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

                best_conf = 0.0
                best_ref_name = None
                # Score each reference using hybrid approach; keep best per frame
                for rp, ref in zip(reference_paths, refs):
                    conf, meta = hybrid_match_score(ref, frame_bgr)
                    if conf > best_conf:
                        best_conf = conf
                        best_ref_name = os.path.basename(rp)

                if best_conf > 0.0:
                    matches.append({
                        "timestamp": float(ts),
                        "confidence": float(best_conf),
                        "reference_image": best_ref_name,
                        "frame_index": j,
                    })
                    matched_frames[j] = frame_bgr
                    max_conf = max(max_conf, float(best_conf))

                if progress_callback and total_frames > 0:
                    progress_callback({
                        "current_video": video_name,
                        "video_index": i,
                        "total_videos": total_videos,
                        "current_frame": j + 1,
                        "total_frames": total_frames,
                        "status": "processing_frames",
                    })

            # Collapse nearby duplicates (preserves frame_index)
            matches = cluster_peaks(matches, window_s=1.0)
            max_conf = max([m.get("confidence", 0.0) for m in matches], default=0.0)

            # Persist preview images for top-N matches by confidence (user-configurable)
            try:
                k = int(top_preview_count)
            except Exception:
                k = 5
            # Clamp to [0, 100]
            k = max(0, min(100, k))
            topN = []
            if k > 0:
                topN = sorted(matches, key=lambda m: m.get("confidence", 0.0), reverse=True)[:k]

            # Remove any existing previews for this video (from earlier saves/runs)
            try:
                vid_prefix = f"{os.path.splitext(video_name)[0]}_"
                for fn in os.listdir(thumb_dir):
                    if fn.startswith(vid_prefix) and (fn.endswith(".jpg") or fn.endswith(".jpeg") or fn.endswith(".png")):
                        try:
                            os.remove(os.path.join(thumb_dir, fn))
                        except Exception:
                            pass
            except Exception:
                pass

            def _save_images(video_name_: str, frame_bgr_: np.ndarray) -> Tuple[str, str]:
                base = f"{os.path.splitext(video_name_)[0]}_{uuid.uuid4().hex}"
                full_name = base + "_full.jpg"
                thumb_name = base + "_thumb.jpg"
                full_path = os.path.join(thumb_dir, full_name)
                thumb_path = os.path.join(thumb_dir, thumb_name)
                # Save full
                cv2.imwrite(full_path, frame_bgr_, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                # Save thumb (width ~ 240px)
                h, w = frame_bgr_.shape[:2]
                target_w = 240
                scale = target_w / max(1, w)
                new_size = (target_w, max(1, int(round(h * scale))))
                thumb_bgr = cv2.resize(frame_bgr_, new_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(thumb_path, thumb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                # Return URLs for browser
                return f"/static/thumbnails/{full_name}", f"/static/thumbnails/{thumb_name}"

            for m in topN:
                fi = m.get("frame_index")
                if fi is None:
                    continue
                frm = matched_frames.get(fi)
                if frm is None:
                    continue
                full_url, thumb_url = _save_images(video_name, frm)
                m["image_full_url"] = full_url
                m["image_thumb_url"] = thumb_url

        except Exception as e:
            matches = [{"error": str(e)}]
            max_conf = 0.0

        results[video_name] = {
            "matches": matches,
            "max_confidence": max_conf,
            "threshold_used": float(confidence_threshold),
            "scanning_mode": scanning_mode,
        }

    return results


if __name__ == "__main__":
    # Simple local sanity check using the thinktank examples (if present)
    # Test with different scanning modes
    ref = ["examples/thinktank/ReferenceImage.png"]
    vids = [
        "examples/thinktank/TT_Positive.mp4",
        "examples/thinktank/TT_Negative.mp4",
    ]
    existing_refs = [p for p in ref if os.path.exists(p)]
    existing_vids = [v for v in vids if os.path.exists(v)]
    if existing_refs and existing_vids:
        # Test with thorough mode (default)
        print("=== THOROUGH MODE ===")
        out = process_videos(existing_refs, existing_vids, frame_interval=1.0, confidence_threshold=0.75, scanning_mode="thorough")
        import json
        print(json.dumps(out, indent=2))
        
        print("\n=== Available Scanning Modes ===")
        for mode_key, mode_config in SCANNING_MODES.items():
            print(f"{mode_key.upper()}: {mode_config['name']}")
            print(f"  Description: {mode_config['description']}")
            print(f"  When to use: {mode_config['when_to_use']}")
            print(f"  Features: {mode_config['orb_params']['nfeatures']}, Scales: {len(mode_config['template_scales'])}, Preprocessing: {mode_config['enable_preprocessing']}")
            print()
    else:
        print("Example files not found; nothing to run.")
