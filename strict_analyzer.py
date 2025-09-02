#!/usr/bin/env python3
"""
Strict Frame Analyzer with improved differentiation between positive and negative results.
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

class StrictFrameAnalyzer:
    def __init__(self):
        """Initialize with CLIP model."""
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        print("CLIP model loaded successfully.")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for CLIP."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image.resize((224, 224), Image.Resampling.LANCZOS)
    
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
    
    def compute_similarity(self, reference_image: Image.Image, frame: Image.Image) -> float:
        """Compute CLIP similarity."""
        ref_img = self.preprocess_image(reference_image)
        frame_img = self.preprocess_image(frame)
        
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
    
    def compute_background_similarity(self, reference_image: Image.Image, 
                                   video_path: str, sample_count: int = 10) -> float:
        """
        Compute average similarity of reference to random frames as a baseline.
        This helps establish what "random" similarity looks like.
        """
        frames = self.extract_frames(video_path, frame_interval=5.0)  # Sample every 5 seconds
        
        # Take a random sample of frames
        if len(frames) > sample_count:
            indices = np.random.choice(len(frames), sample_count, replace=False)
            sampled_frames = [frames[i][0] for i in indices]
        else:
            sampled_frames = [frame for frame, _ in frames]
        
        if not sampled_frames:
            return 0.0
        
        similarities = []
        for frame in sampled_frames:
            sim = self.compute_similarity(reference_image, frame)
            similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def detect_reference_in_video(self, reference_path: str, video_path: str, 
                                frame_interval: float = 0.5,
                                background_video_path: str = None) -> Dict:
        """
        Strict detection with background comparison.
        """
        # Load reference image
        reference_image = Image.open(reference_path)
        
        # Compute background similarity if provided
        background_similarity = 0.0
        if background_video_path and os.path.exists(background_video_path):
            try:
                background_similarity = self.compute_background_similarity(
                    reference_image, background_video_path
                )
                print(f"Background similarity baseline: {background_similarity:.3f}")
            except Exception as e:
                print(f"Could not compute background similarity: {e}")
        
        # Extract frames
        frames = self.extract_frames(video_path, frame_interval)
        
        # Store results
        similarities = []
        timestamps = []
        
        print(f"Processing {len(frames)} frames...")
        
        # Compare each frame with reference image
        for i, (frame, timestamp) in enumerate(frames):
            similarity = self.compute_similarity(reference_image, frame)
            similarities.append(similarity)
            timestamps.append(timestamp)
            
            # Progress indicator
            if (i + 1) % 50 == 0 or (i + 1) == len(frames):
                max_sim = max(similarities) if similarities else 0
                print(f"Processed {i + 1}/{len(frames)} frames (max similarity: {max_sim:.3f})")
        
        # Convert to numpy arrays
        similarities = np.array(similarities)
        
        # Find maximum similarity
        max_similarity = float(np.max(similarities)) if len(similarities) > 0 else 0.0
        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        
        # Enhanced detection with multiple criteria
        # 1. Absolute threshold
        absolute_threshold = 0.32  # Increased threshold
        
        # 2. Relative to background threshold
        relative_threshold = background_similarity + 0.15 if background_similarity > 0 else 0
        
        # 3. Statistical significance
        if std_sim > 0:
            z_score = (max_similarity - mean_sim) / std_sim
        else:
            z_score = 0 if max_similarity == mean_sim else float('inf')
        
        # 4. Cluster detection
        high_frames = similarities > max(0.3, mean_sim + std_sim)
        consecutive_high = 0
        max_consecutive = 0
        
        for is_high in high_frames:
            if is_high:
                consecutive_high += 1
                max_consecutive = max(max_consecutive, consecutive_high)
            else:
                consecutive_high = 0
        
        has_cluster = max_consecutive >= 5  # Increased requirement
        
        # 5. Minimum high frames count
        min_high_frames = 10
        num_high_frames = np.sum(high_frames)
        enough_high_frames = num_high_frames >= min_high_frames
        
        # Final determination - ALL criteria must be met
        meets_absolute = max_similarity >= absolute_threshold
        meets_relative = max_similarity >= relative_threshold
        significant_peak = z_score > 3.0  # Stricter statistical requirement
        
        found = (
            meets_absolute and 
            meets_relative and 
            significant_peak and 
            has_cluster and 
            enough_high_frames
        )
        
        confidence = max_similarity if found else 0.0
        
        # Find matches
        matches = []
        if found:
            match_threshold = max(0.3, mean_sim + 0.5 * std_sim)
            match_indices = np.where(similarities >= match_threshold)[0]
            
            for idx in match_indices:
                matches.append({
                    "timestamp": float(timestamps[idx]),
                    "similarity": float(similarities[idx])
                })
        
        return {
            "found": found,
            "confidence": confidence,
            "max_similarity": max_similarity,
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "background_similarity": background_similarity,
            "z_score": z_score,
            "matches": matches,
            "total_frames_processed": len(frames),
            "criteria": {
                "meets_absolute": meets_absolute,
                "meets_relative": meets_relative,
                "significant_peak": significant_peak,
                "has_cluster": has_cluster,
                "enough_high_frames": enough_high_frames,
                "num_high_frames": int(num_high_frames)
            }
        }

def main():
    """Main function for testing the strict analyzer."""
    # Initialize analyzer
    analyzer = StrictFrameAnalyzer()
    
    # Define paths
    base_path = "examples/thinktank"
    reference_path = os.path.join(base_path, "ReferenceImage.png")
    
    # Test with available files
    test_files = []
    for file in os.listdir(base_path):
        if file.endswith('.mp4'):
            test_files.append(os.path.join(base_path, file))
    
    print("\n" + "="*70)
    print("STRICT FRAME FINDER - TESTING WITH EXAMPLE FILES")
    print("="*70)
    
    results = {}
    
    for i, video_path in enumerate(test_files):
        print(f"\nTesting with video: {os.path.basename(video_path)}")
        print("-" * 50)
        
        if os.path.exists(video_path):
            # Use the other video as background reference for comparison
            background_video = test_files[1-i] if len(test_files) > 1 else None
            
            result = analyzer.detect_reference_in_video(
                reference_path, 
                video_path, 
                frame_interval=1.0,  # Faster processing
                background_video_path=background_video
            )
            
            results[os.path.basename(video_path)] = result
            
            print(f"Reference found: {result['found']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Max similarity: {result['max_similarity']:.3f}")
            print(f"Mean similarity: {result['mean_similarity']:.3f}")
            print(f"Std similarity: {result['std_similarity']:.3f}")
            print(f"Background similarity: {result['background_similarity']:.3f}")
            print(f"Z-score: {result['z_score']:.3f}")
            print(f"Matches found: {len(result['matches'])}")
            print(f"Total frames processed: {result['total_frames_processed']}")
            
            # Print criteria
            criteria = result.get('criteria', {})
            print(f"Criteria met:")
            print(f"  Meets absolute threshold: {criteria.get('meets_absolute', 'N/A')}")
            print(f"  Meets relative threshold: {criteria.get('meets_relative', 'N/A')}")
            print(f"  Significant peak: {criteria.get('significant_peak', 'N/A')}")
            print(f"  Has cluster: {criteria.get('has_cluster', 'N/A')}")
            print(f"  Enough high frames: {criteria.get('enough_high_frames', 'N/A')} ({criteria.get('num_high_frames', 0)} high frames)")
            
            if result['matches']:
                print("Top matches:")
                for i, match in enumerate(result['matches'][:3]):  # Show top 3
                    print(f"  {i+1}. Time: {match['timestamp']:.1f}s, Similarity: {match['similarity']:.3f}")
        else:
            print("Video file not found!")
    
    # Compare results if we have multiple videos
    if len(results) >= 2:
        print("\n" + "="*70)
        print("COMPARISON OF RESULTS")
        print("="*70)
        
        items = list(results.items())
        video1_name, video1_result = items[0]
        video2_name, video2_result = items[1]
        
        print(f"\n{video1_name} vs {video2_name}:")
        print(f"  Max similarity: {video1_result['max_similarity']:.3f} vs {video2_result['max_similarity']:.3f}")
        print(f"  Found: {video1_result['found']} vs {video2_result['found']}")
        print(f"  Confidence: {video1_result['confidence']:.3f} vs {video2_result['confidence']:.3f}")
        
        diff = abs(video1_result['max_similarity'] - video2_result['max_similarity'])
        print(f"  Difference: {diff:.3f}")
        
        if diff > 0.1:
            print("  Result: Clear distinction between videos")
        elif diff > 0.05:
            print("  Result: Moderate distinction between videos")
        else:
            print("  Result: Minimal distinction between videos - may be challenging to differentiate")

if __name__ == "__main__":
    main()