#!/usr/bin/env python3
"""
Frame Analyzer for finding reference images in videos.
This implementation focuses on precision to minimize false positives.
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

# Model configuration
MODEL_NAME = "openai/clip-vit-base-patch32"

class FrameAnalyzer:
    def __init__(self):
        """Initialize the FrameAnalyzer with CLIP model."""
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        print("CLIP model loaded successfully.")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess an image for better matching.
        Convert to RGB and resize to a standard size.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to a standard size for consistency
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        return image
    
    def extract_frames(self, video_path: str, frame_interval: float = 0.5) -> List[Tuple[Image.Image, float]]:
        """
        Extract frames from video at regular intervals.
        
        Args:
            video_path: Path to the video file
            frame_interval: Interval in seconds between extracted frames
            
        Returns:
            List of tuples (PIL Image, timestamp in seconds)
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default fallback
            
        frame_step = max(1, int(fps * frame_interval))
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_step == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                timestamp = frame_idx / fps
                frames.append((pil_image, timestamp))
                
            frame_idx += 1
            
        cap.release()
        return frames
    
    def compute_similarity(self, reference_image: Image.Image, frame: Image.Image) -> float:
        """
        Compute similarity between reference image and frame using CLIP.
        
        Args:
            reference_image: Reference image to match
            frame: Frame to compare against
            
        Returns:
            Similarity score (cosine similarity)
        """
        # Preprocess images
        ref_img = self.preprocess_image(reference_image)
        frame_img = self.preprocess_image(frame)
        
        # Process images with CLIP processor
        inputs = self.processor(
            images=[ref_img, frame_img],
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        
        # Get image features
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            # Normalize features
            features = torch.nn.functional.normalize(features, dim=-1)
            
        # Compute cosine similarity
        similarity = torch.cosine_similarity(features[0].unsqueeze(0), features[1].unsqueeze(0))
        return similarity.item()
    
    def detect_reference_in_video(self, reference_path: str, video_path: str, 
                                frame_interval: float = 0.5) -> Dict:
        """
        Detect if reference image appears in video with strict false positive prevention.
        
        Args:
            reference_path: Path to reference image
            video_path: Path to video file
            frame_interval: Interval in seconds between frames
            
        Returns:
            Dictionary with detection results
        """
        # Load reference image
        reference_image = Image.open(reference_path)
        
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
        
        # Convert to numpy arrays for easier processing
        similarities = np.array(similarities)
        
        # Find the maximum similarity
        max_similarity = float(np.max(similarities)) if len(similarities) > 0 else 0.0
        
        # Use a strict detection approach:
        # 1. Require a high minimum threshold (0.4)
        # 2. Require a clear peak above background noise
        # 3. Use statistical analysis but be reasonable about variation expectations
        # 4. Require sufficient variation to distinguish signal from noise
        
        min_threshold = 0.45  # Increased threshold
        
        # If max similarity is below our minimum threshold, it's definitely not found
        if max_similarity < min_threshold:
            return {
                "found": False,
                "confidence": 0.0,
                "max_similarity": max_similarity,
                "matches": [],
                "total_frames_processed": len(frames)
            }
        
        # Calculate statistics
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Use a strict approach:
        # - Require max similarity to be significantly above mean
        # - Require sufficient variation to distinguish signal from noise
        # - Require a cluster of high similarity frames
        
        significant_peak = max_similarity > mean_sim + 3 * std_sim  # Increased threshold
        sufficient_variation = std_sim > 0.03  # Reasonable variation threshold
        
        # Additional check: require a cluster of high similarity frames
        # A single high frame among low ones is likely a false positive
        high_frames = similarities > max(0.4, mean_sim + 2 * std_sim)
        consecutive_high = 0
        max_consecutive = 0
        
        for is_high in high_frames:
            if is_high:
                consecutive_high += 1
                max_consecutive = max(max_consecutive, consecutive_high)
            else:
                consecutive_high = 0
        
        # Require at least 3 consecutive high frames to reduce false positives
        has_cluster = max_consecutive >= 3
        
        # Additional check: require a minimum number of high frames
        min_high_frames = 5
        num_high_frames = np.sum(high_frames)
        enough_high_frames = num_high_frames >= min_high_frames
        
        # Final determination - be strict to avoid false positives
        found = (
            max_similarity >= min_threshold and 
            significant_peak and 
            (sufficient_variation or max_similarity > 0.75) and  # Either sufficient variation or very high similarity
            has_cluster and
            enough_high_frames  # Require enough high frames
        )
        
        # Confidence is the maximum similarity if found, otherwise 0
        confidence = max_similarity if found else 0.0
        
        # Find detailed match information if found
        matches = []
        if found:
            # Use a threshold that's still strict but captures the matches
            match_threshold = max(0.4, mean_sim + 1.5 * std_sim)
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
            "mean_similarity": float(mean_sim),
            "std_similarity": float(std_sim),
            "matches": matches,
            "total_frames_processed": len(frames),
            "stats": {
                "significant_peak": significant_peak,
                "sufficient_variation": sufficient_variation,
                "has_cluster": has_cluster,
                "enough_high_frames": enough_high_frames,
                "num_high_frames": int(num_high_frames)
            }
        }

def main():
    """Main function for testing the analyzer with the provided examples."""
    # Initialize analyzer
    analyzer = FrameAnalyzer()
    
    # Define paths
    base_path = "examples/thinktank"
    reference_path = os.path.join(base_path, "ReferenceImage.png")
    positive_video = os.path.join(base_path, "TT_Positive.mp4")
    negative_video = os.path.join(base_path, "TT_Negative.mp4")
    
    print("\n" + "="*60)
    print("FRAME FINDER - TESTING WITH EXAMPLE FILES")
    print("="*60)
    
    # Test with positive video
    print(f"\nTesting with positive video: {positive_video}")
    print("-" * 40)
    
    if os.path.exists(positive_video):
        result_pos = analyzer.detect_reference_in_video(
            reference_path, 
            positive_video, 
            frame_interval=0.5
        )
        
        print(f"Reference found: {result_pos['found']}")
        print(f"Confidence: {result_pos['confidence']:.3f}")
        print(f"Max similarity: {result_pos['max_similarity']:.3f}")
        print(f"Mean similarity: {result_pos['mean_similarity']:.3f}")
        print(f"Std similarity: {result_pos['std_similarity']:.3f}")
        print(f"Matches found: {len(result_pos['matches'])}")
        print(f"Total frames processed: {result_pos['total_frames_processed']}")
        
        # Print detection criteria
        stats = result_pos.get('stats', {})
        print(f"Significant peak: {stats.get('significant_peak', 'N/A')}")
        print(f"Sufficient variation: {stats.get('sufficient_variation', 'N/A')}")
        print(f"Has cluster: {stats.get('has_cluster', 'N/A')}")
        print(f"Enough high frames: {stats.get('enough_high_frames', 'N/A')} ({stats.get('num_high_frames', 0)} high frames)")
        
        if result_pos['matches']:
            print("Match timestamps (s):", [f"{m['timestamp']:.1f}" for m in result_pos['matches'][:5]])  # Show first 5
    else:
        print("Positive video file not found!")
    
    # Test with negative video
    print(f"\nTesting with negative video: {negative_video}")
    print("-" * 40)
    
    if os.path.exists(negative_video):
        result_neg = analyzer.detect_reference_in_video(
            reference_path, 
            negative_video, 
            frame_interval=0.5
        )
        
        print(f"Reference found: {result_neg['found']}")
        print(f"Confidence: {result_neg['confidence']:.3f}")
        print(f"Max similarity: {result_neg['max_similarity']:.3f}")
        print(f"Mean similarity: {result_neg['mean_similarity']:.3f}")
        print(f"Std similarity: {result_neg['std_similarity']:.3f}")
        print(f"Matches found: {len(result_neg['matches'])}")
        print(f"Total frames processed: {result_neg['total_frames_processed']}")
        
        # Print detection criteria
        stats = result_neg.get('stats', {})
        print(f"Significant peak: {stats.get('significant_peak', 'N/A')}")
        print(f"Sufficient variation: {stats.get('sufficient_variation', 'N/A')}")
        print(f"Has cluster: {stats.get('has_cluster', 'N/A')}")
        print(f"Enough high frames: {stats.get('enough_high_frames', 'N/A')} ({stats.get('num_high_frames', 0)} high frames)")
    else:
        print("Negative video file not found!")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()