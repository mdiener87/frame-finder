#!/usr/bin/env python3
"""
Multi-strategy Frame Analyzer that combines multiple approaches for better accuracy.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from typing import List, Tuple, Dict, Optional
import os

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

class MultiStrategyAnalyzer:
    def __init__(self):
        """Initialize with both CLIP and traditional computer vision approaches."""
        print("Loading CLIP model...")
        # Using both base and large models for comparison
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP models loaded successfully.")
        
        # Initialize ORB detector for traditional feature matching
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def preprocess_for_clip(self, image: Image.Image) -> Image.Image:
        """Preprocess image for CLIP."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image.resize((224, 224), Image.Resampling.LANCZOS)
    
    def preprocess_for_cv(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for traditional computer vision."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Convert to grayscale for ORB
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    def extract_frames(self, video_path: str, frame_interval: float = 0.5) -> List[Tuple[Image.Image, float]]:
        """Extract frames from video."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(1, int(fps * frame_interval))
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_step == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                timestamp = frame_idx / fps
                frames.append((pil_image, timestamp))
                
            frame_idx += 1
            
        cap.release()
        return frames
    
    def clip_similarity(self, reference_image: Image.Image, frame: Image.Image) -> float:
        """Compute CLIP similarity."""
        ref_img = self.preprocess_for_clip(reference_image)
        frame_img = self.preprocess_for_clip(frame)
        
        inputs = self.clip_processor(
            images=[ref_img, frame_img],
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)
            
        similarity = torch.cosine_similarity(features[0].unsqueeze(0), features[1].unsqueeze(0))
        return float(similarity.item())
    
    def orb_similarity(self, reference_image: Image.Image, frame: Image.Image) -> float:
        """Compute ORB feature matching similarity."""
        ref_gray = self.preprocess_for_cv(reference_image)
        frame_gray = self.preprocess_for_cv(frame)
        
        # Detect and compute keypoints
        kp1, des1 = self.orb.detectAndCompute(ref_gray, None)
        kp2, des2 = self.orb.detectAndCompute(frame_gray, None)
        
        if des1 is None or des2 is None:
            return 0.0
        
        # Match features
        matches = self.bf.match(des1, des2)
        
        if len(matches) == 0:
            return 0.0
        
        # Calculate similarity based on good matches
        # Using distance as inverse of similarity (lower distance = higher similarity)
        distances = [m.distance for m in matches]
        avg_distance = np.mean(distances)
        
        # Convert distance to similarity (arbitrary scaling)
        # ORB distances are typically 0-255, so we invert and normalize
        similarity = max(0, 1 - (avg_distance / 255.0))
        return float(similarity)
    
    def combined_similarity(self, reference_image: Image.Image, frame: Image.Image) -> Dict[str, float]:
        """Compute combined similarity using multiple approaches."""
        clip_sim = self.clip_similarity(reference_image, frame)
        orb_sim = self.orb_similarity(reference_image, frame)
        
        # Weighted combination (adjustable weights)
        clip_weight = 0.7
        orb_weight = 0.3
        combined = clip_sim * clip_weight + orb_sim * orb_weight
        
        return {
            'clip': clip_sim,
            'orb': orb_sim,
            'combined': combined
        }
    
    def detect_reference_in_video(self, reference_path: str, video_path: str, 
                                frame_interval: float = 0.5) -> Dict:
        """
        Enhanced detection using multiple strategies.
        """
        # Load reference image
        reference_image = Image.open(reference_path)
        
        # Extract frames
        frames = self.extract_frames(video_path, frame_interval)
        
        # Store results
        clip_similarities = []
        orb_similarities = []
        combined_similarities = []
        timestamps = []
        
        print(f"Processing {len(frames)} frames...")
        
        # Compare each frame with reference image using multiple strategies
        for i, (frame, timestamp) in enumerate(frames):
            similarities = self.combined_similarity(reference_image, frame)
            
            clip_similarities.append(similarities['clip'])
            orb_similarities.append(similarities['orb'])
            combined_similarities.append(similarities['combined'])
            timestamps.append(timestamp)
            
            # Progress indicator
            if (i + 1) % 50 == 0 or (i + 1) == len(frames):
                max_combined = max(combined_similarities) if combined_similarities else 0
                print(f"Processed {i + 1}/{len(frames)} frames (max combined similarity: {max_combined:.3f})")
        
        # Convert to numpy arrays
        clip_similarities = np.array(clip_similarities)
        orb_similarities = np.array(orb_similarities)
        combined_similarities = np.array(combined_similarities)
        
        # Find maximum similarities
        max_clip = float(np.max(clip_similarities)) if len(clip_similarities) > 0 else 0.0
        max_orb = float(np.max(orb_similarities)) if len(orb_similarities) > 0 else 0.0
        max_combined = float(np.max(combined_similarities)) if len(combined_similarities) > 0 else 0.0
        
        mean_combined = float(np.mean(combined_similarities))
        std_combined = float(np.std(combined_similarities))
        
        # Enhanced detection algorithm
        min_threshold = 0.3  # Combined similarity threshold
        
        if max_combined < min_threshold:
            return {
                "found": False,
                "confidence": 0.0,
                "max_similarities": {
                    "clip": max_clip,
                    "orb": max_orb,
                    "combined": max_combined
                },
                "mean_similarity": mean_combined,
                "std_similarity": std_combined,
                "matches": [],
                "total_frames_processed": len(frames)
            }
        
        # Statistical approach
        if std_combined > 0:
            z_score = (max_combined - mean_combined) / std_combined
        else:
            z_score = 0 if max_combined == mean_combined else float('inf')
        
        # Require significant statistical difference
        significant_peak = z_score > 2.5  # Lowered threshold for combined approach
        
        # Cluster detection
        high_frames = combined_similarities > max(0.25, mean_combined + 0.5 * std_combined)
        consecutive_high = 0
        max_consecutive = 0
        
        for is_high in high_frames:
            if is_high:
                consecutive_high += 1
                max_consecutive = max(max_consecutive, consecutive_high)
            else:
                consecutive_high = 0
        
        has_cluster = max_consecutive >= 3
        
        # Final determination
        found = max_combined >= min_threshold and significant_peak and has_cluster
        confidence = max_combined if found else 0.0
        
        # Find matches
        matches = []
        if found:
            match_threshold = max(0.25, mean_combined + 0.25 * std_combined)
            match_indices = np.where(combined_similarities >= match_threshold)[0]
            
            for idx in match_indices:
                matches.append({
                    "timestamp": float(timestamps[idx]),
                    "similarity": float(combined_similarities[idx]),
                    "clip_similarity": float(clip_similarities[idx]),
                    "orb_similarity": float(orb_similarities[idx])
                })
        
        return {
            "found": found,
            "confidence": confidence,
            "max_similarities": {
                "clip": max_clip,
                "orb": max_orb,
                "combined": max_combined
            },
            "mean_similarity": mean_combined,
            "std_similarity": std_combined,
            "z_score": z_score,
            "matches": matches,
            "total_frames_processed": len(frames),
            "stats": {
                "significant_peak": significant_peak,
                "has_cluster": has_cluster
            }
        }

def main():
    """Main function for testing the multi-strategy analyzer."""
    # Initialize analyzer
    analyzer = MultiStrategyAnalyzer()
    
    # Define paths
    base_path = "examples/thinktank"
    reference_path = os.path.join(base_path, "ReferenceImage.png")
    
    # Test with available files
    test_files = []
    for file in os.listdir(base_path):
        if file.endswith('.mp4'):
            test_files.append(os.path.join(base_path, file))
    
    print("\n" + "="*70)
    print("MULTI-STRATEGY FRAME FINDER - TESTING WITH EXAMPLE FILES")
    print("="*70)
    
    for video_path in test_files:
        print(f"\nTesting with video: {os.path.basename(video_path)}")
        print("-" * 50)
        
        if os.path.exists(video_path):
            result = analyzer.detect_reference_in_video(
                reference_path, 
                video_path, 
                frame_interval=1.0  # Faster processing for demo
            )
            
            print(f"Reference found: {result['found']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Max Similarities:")
            max_sims = result['max_similarities']
            print(f"  CLIP: {max_sims['clip']:.3f}")
            print(f"  ORB: {max_sims['orb']:.3f}")
            print(f"  Combined: {max_sims['combined']:.3f}")
            print(f"Mean similarity: {result['mean_similarity']:.3f}")
            print(f"Std similarity: {result['std_similarity']:.3f}")
            print(f"Z-score: {result['z_score']:.3f}")
            print(f"Matches found: {len(result['matches'])}")
            print(f"Total frames processed: {result['total_frames_processed']}")
            
            # Print detection criteria
            stats = result.get('stats', {})
            print(f"Significant peak: {stats.get('significant_peak', 'N/A')}")
            print(f"Has cluster: {stats.get('has_cluster', 'N/A')}")
            
            if result['matches']:
                print("Top matches:")
                for i, match in enumerate(result['matches'][:3]):  # Show top 3
                    print(f"  {i+1}. Time: {match['timestamp']:.1f}s, "
                          f"Combined: {match['similarity']:.3f}, "
                          f"CLIP: {match['clip_similarity']:.3f}, "
                          f"ORB: {match['orb_similarity']:.3f}")
        else:
            print("Video file not found!")

if __name__ == "__main__":
    main()