# analyzer.py
import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

# ==============================
# MODE SWITCH (baseline | advanced)
# ==============================
MODE = "baseline"   # <-- change to "advanced" for precision mode

# ------------------------------
# Baseline defaults (raw CLIP cosine)
# ------------------------------
BASELINE_FRAME_INTERVAL = 0.5      # seconds (2 fps)
BASELINE_THRESHOLD = 0.30          # raw cosine on ViT-B/32 ~0.28–0.40 for good matches
BASELINE_CLUSTER_WINDOW_S = 1.0

# ------------------------------
# Advanced defaults (delta scoring, adaptive threshold)
# ------------------------------
ADV_FRAME_INTERVAL = 0.5           # coarse pass; refine is optional
ADV_CONFIDENCE_THRESHOLD = None    # None => adaptive (μ + zσ)
ADV_ADAPTIVE_Z = 3.5               # stricter = fewer FPs
ADV_CLUSTER_WINDOW_S = 1.0
ADV_USE_ORB_GATE = True            # helps kill hull FPs cheaply
ADV_ORB_MIN_GOOD = 18              # tune 14–24 depending on clip/ref
ADV_ORB_DIST_PERCENTILE = 35       # 30–45 typically works
ADV_USE_NORMALIZATION = True       # CLAHE on L channel in LAB
ADV_NEGATIVE_SCORING = True        # enable delta scoring if negative refs provided

# ==============================
# Model / device
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"  # try "openai/clip-vit-large-patch14" if you have VRAM

model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_ID)

# ==============================
# Allowed extensions
# ==============================
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==============================
# Image normalization (used in advanced mode)
# ==============================
def normalize_pil(img: Image.Image) -> Image.Image:
    """CLAHE on L channel to stabilize lighting/compression artifacts."""
    arr = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


# ==============================
# Frame extraction
# ==============================
def extract_frames(video_path: str, interval: float = 1.0):
    """Return list of (PIL.Image, timestamp_seconds) sampled by modulo frame step."""
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
            frames.append((Image.fromarray(rgb), idx / fps))
        idx += 1
    cap.release()
    return frames


# ==============================
# CLIP encoding helpers (batched)
# ==============================
@torch.no_grad()
def encode_images_batched(pil_list, batch_size: int = 32, normalize: bool = False):
    """Return L2-normalized image features tensor of shape (N, D) on DEVICE."""
    feats = []
    for i in range(0, len(pil_list), batch_size):
        batch = pil_list[i:i + batch_size]
        if normalize:
            batch = [normalize_pil(im) for im in batch]
        inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
        f = model.get_image_features(**inputs)
        f = torch.nn.functional.normalize(f, dim=-1)
        feats.append(f)
    if len(feats) == 0:
        return torch.empty((0, model.visual_projection.out_features), device=DEVICE)
    return torch.cat(feats, dim=0)


# ==============================
# Reference embeddings cache
# ==============================
class EmbeddingsCache:
    def __init__(self, image_paths, normalize=False):
        self.paths = list(image_paths or [])
        self.normalize = normalize
        self.pils = [Image.open(p) for p in self.paths]
        self.feats = encode_images_batched(self.pils, normalize=self.normalize)

    @property
    def empty(self):
        return self.feats.numel() == 0

    def best_sim(self, frame_feats: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Return (best_sim_values, best_indices) for each frame feat."""
        # frame_feats: (F, D); self.feats: (R, D)
        sims = frame_feats @ self.feats.T  # (F, R)
        best_vals, best_idx = sims.max(dim=1)
        return best_vals, best_idx


# ==============================
# Optional ORB gate (cheap geometric check)
# ==============================
class ORBGate:
    def __init__(self, ref_pils, min_good=18, dist_percentile=35):
        self.min_good = min_good
        self.dist_percentile = dist_percentile
        # Precompute descriptors for refs
        self.orb = cv2.ORB_create()
        self.ref_descs = []
        for im in ref_pils:
            g = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
            k, d = self.orb.detectAndCompute(g, None)
            self.ref_descs.append(d)

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def passes(self, frame_pil: Image.Image) -> bool:
        g = cv2.cvtColor(np.array(frame_pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
        kf, df = self.orb.detectAndCompute(g, None)
        if df is None or len(df) == 0:
            return False
        for dr in self.ref_descs:
            if dr is None or len(dr) == 0:
                continue
            matches = self.bf.match(df, dr)
            if not matches:
                continue
            dists = np.array([m.distance for m in matches], dtype=np.float32)
            cutoff = np.percentile(dists, self.dist_percentile)
            good = int((dists <= cutoff).sum())
            if good >= self.min_good:
                return True
        return False


# ==============================
# Temporal clustering
# ==============================
def cluster_peaks(matches, window_s: float = 1.0):
    """Matches: list of dicts with 'timestamp' and 'confidence'."""
    if not matches:
        return []
    matches.sort(key=lambda m: m["timestamp"])
    out, cur = [], [matches[0]]
    for m in matches[1:]:
        if m["timestamp"] - cur[-1]["timestamp"] <= window_s:
            cur.append(m)
        else:
            out.append(max(cur, key=lambda x: x["confidence"]))
            cur = [m]
    out.append(max(cur, key=lambda x: x["confidence"]))
    return out


# ==============================
# Adaptive threshold for advanced mode
# ==============================
def adaptive_threshold_from_frames(frames, ref_cache: EmbeddingsCache,
                                   neg_cache: EmbeddingsCache | None,
                                   normalize: bool, stride: int = 10, z: float = 3.5):
    sample_pils = [frames[i][0] for i in range(0, len(frames), max(1, stride))]
    if len(sample_pils) == 0:
        return 1.0  # effectively disable
    frame_feats = encode_images_batched(sample_pils, normalize=normalize)
    # raw best sim to refs
    ref_best, _ = ref_cache.best_sim(frame_feats)
    scores = ref_best
    if neg_cache and not neg_cache.empty and ADV_NEGATIVE_SCORING:
        # subtract max sim to negatives
        neg_best, _ = neg_cache.best_sim(frame_feats)
        scores = ref_best - neg_best
    arr = scores.float().detach().cpu().numpy()
    mu, sd = float(arr.mean()), float(arr.std() + 1e-6)
    return mu + z * sd


# ==============================
# Public API: process_videos
# ==============================
def process_videos(reference_paths, video_paths, frame_interval=1.0,
                   confidence_threshold=0.5, negative_paths=None, progress_callback=None):
    """
    Args:
        reference_paths (list[str]): paths to positive reference images
        video_paths     (list[str]): paths to mp4 files
        frame_interval  (float)    : seconds between frames
        confidence_threshold (float|None): if None in advanced mode, uses adaptive
        negative_paths (list[str]|None): optional background negatives (advanced)
        progress_callback (callable) : callback function to report progress (optional)
    Returns:
        dict[str, list[dict]]: { video_name: [ {timestamp, confidence, reference_image} ... ] }
    """
    # Choose mode config
    if MODE == "baseline":
        interval = frame_interval if frame_interval else BASELINE_FRAME_INTERVAL
        thr = confidence_threshold if confidence_threshold is not None else BASELINE_THRESHOLD
        normalize = False
        use_neg = False
        use_orb = False
        cluster_window = BASELINE_CLUSTER_WINDOW_S
    elif MODE == "advanced":
        interval = frame_interval if frame_interval else ADV_FRAME_INTERVAL
        thr = confidence_threshold  # may be None => adaptive
        normalize = ADV_USE_NORMALIZATION
        use_neg = ADV_NEGATIVE_SCORING
        use_orb = ADV_USE_ORB_GATE
        cluster_window = ADV_CLUSTER_WINDOW_S
    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    # Build caches
    ref_cache = EmbeddingsCache(reference_paths, normalize=normalize)
    neg_cache = EmbeddingsCache(negative_paths, normalize=normalize) if (negative_paths and use_neg) else None

    # ORB gate (advanced)
    orb_gate = ORBGate(ref_cache.pils, ADV_ORB_MIN_GOOD, ADV_ORB_DIST_PERCENTILE) if (MODE == "advanced" and use_orb) else None

    results = {}

    # Calculate total number of videos for progress tracking
    total_videos = len(video_paths)

    for i, video_path in enumerate(video_paths):
        video_name = os.path.basename(video_path)
        matches = []
        try:
            # Report which video is being processed
            if progress_callback:
                progress_callback({
                    'current_video': video_name,
                    'video_index': i,
                    'total_videos': total_videos,
                    'status': 'processing_video'
                })
            
            frames = extract_frames(video_path, interval=interval)

            # adaptive threshold (advanced, if None supplied)
            vid_thr = thr
            if MODE == "advanced" and thr is None:
                vid_thr = adaptive_threshold_from_frames(
                    frames, ref_cache, neg_cache, normalize=normalize,
                    stride=10, z=ADV_ADAPTIVE_Z
                )

            # batch encode frames (respect normalization mode)
            frame_pils = [f for (f, _) in frames]
            frame_ts = [t for (_, t) in frames]

            # optional ORB gating
            if orb_gate is not None:
                gated = [(f, t) for (f, t) in frames if orb_gate.passes(f)]
                if not gated:
                    results[video_name] = []
                    continue
                frame_pils = [f for (f, _) in gated]
                frame_ts = [t for (_, t) in gated]

            if len(frame_pils) == 0:
                results[video_name] = []
                continue

            frame_feats = encode_images_batched(frame_pils, normalize=normalize)

            # Best sim to refs (top-1 per frame)
            ref_best, ref_idx = ref_cache.best_sim(frame_feats)
            scores = ref_best

            # Delta scoring with negatives (advanced)
            if MODE == "advanced" and neg_cache and not neg_cache.empty and use_neg:
                neg_best, _ = neg_cache.best_sim(frame_feats)
                scores = ref_best - neg_best

            # Collect hits (top-1 per frame only)
            scores_np = scores.float().detach().cpu().numpy()
            ref_idx_np = ref_idx.detach().cpu().numpy()
            total_frames = len(frame_ts)
            for j, (ts, s, ridx) in enumerate(zip(frame_ts, scores_np, ref_idx_np)):
                matches.append({
                    "timestamp": float(ts),
                    "confidence": float(s),
                    "reference_image": os.path.basename(reference_paths[ridx]) if len(reference_paths) else None
                })
                
                # Report progress within the video
                if progress_callback and total_frames > 0:
                    progress_callback({
                        'current_video': video_name,
                        'video_index': i,
                        'total_videos': total_videos,
                        'current_frame': j,
                        'total_frames': total_frames,
                        'status': 'processing_frames'
                    })

            # Temporal clustering
            matches = cluster_peaks(matches, window_s=cluster_window)
            
            # Calculate max confidence for UI display
            max_confidence = max([m["confidence"] for m in matches], default=0.0)

        except Exception as e:
            matches = [{"error": str(e)}]
            max_confidence = 0.0

        results[video_name] = {
            "matches": matches,
            "max_confidence": max_confidence,
            "threshold_used": vid_thr
        }

    return results
