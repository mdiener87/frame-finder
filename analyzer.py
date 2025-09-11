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
    def __init__(self, path: str, max_size: int = 1024):
        self.path = path
        self.image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.image_bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        h, w = self.image_bgr.shape[:2]
        scale = min(1.0, max_size / max(h, w))
        if scale < 0.999:
            self.image_bgr = cv2.resize(self.image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        self.gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)

        # ORB is fast and robust enough; increase features for better recall.
        self.detector = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8)
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
            # Create CUDA ORB only once per call (fast enough), or could cache globally
            orb_gpu = cv2.cuda_ORB_create(nfeatures=3000)
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

    if len(good) < 10:
        return 0.0, {"reason": "too_few_good_matches", "good": len(good)}

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
                   progress_callback=None) -> Dict[str, Any]:
    """
    Args:
        reference_paths: paths to positive reference images
        video_paths    : paths to mp4 files
        frame_interval : seconds between frames extracted
        confidence_threshold: retained for UI compatibility; analyzer returns all matches
        progress_callback: optional callable for UI progress updates

    Returns:
        { video_name: { matches: [...], max_confidence: float, threshold_used: float } }
    """
    # Precompute reference features once
    refs: List[RefFeatures] = [RefFeatures(p) for p in (reference_paths or [])]

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
                # Score each reference; keep best per frame
                for rp, ref in zip(reference_paths, refs):
                    conf, _meta = match_and_score(ref, frame_bgr)
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

            # Persist preview images for top 5 matches by confidence
            top5 = sorted(matches, key=lambda m: m.get("confidence", 0.0), reverse=True)[:5]

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

            for m in top5:
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
        }

    return results


if __name__ == "__main__":
    # Simple local sanity check using the thinktank examples (if present)
    ref = ["examples/thinktank/ReferenceImage.png"]
    vids = [
        "examples/thinktank/TT_Positive.mp4",
        "examples/thinktank/TT_Negative.mp4",
    ]
    existing_refs = [p for p in ref if os.path.exists(p)]
    existing_vids = [v for v in vids if os.path.exists(v)]
    if existing_refs and existing_vids:
        out = process_videos(existing_refs, existing_vids, frame_interval=1.0, confidence_threshold=0.75)
        import json
        print(json.dumps(out, indent=2))
    else:
        print("Example files not found; nothing to run.")
