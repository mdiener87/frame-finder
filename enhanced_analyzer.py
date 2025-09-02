#!/usr/bin/env python3
"""
Enhanced Frame Analyzer with improved detection logic.
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
MODEL_NAME = "openai/clip-vit-large-patch14"  # Using a larger model for better accuracy

class EnhancedFrameAnalyzer:
    def __init__(self):
        """Initialize the EnhancedFrameAnalyzer with CLIP model."""
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        print("CLIP model loaded successfully.")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Enhanced preprocessing with normalization.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) 
        # to normalize lighting conditions
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL Image and resize
        enhanced_image = Image.fromarray(img_array)
        enhanced_image = enhanced_image.resize((224, 224), Image.Resampling.LANCZOS)
        return enhanced_image
    
    def extract_frames(self, video_path: str, frame_interval: float = 0.5) -> List[Tuple[Image.Image, float]]:
        """
        Extract frames from video at regular intervals.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
            
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
    
    def compute_similarity_with_negative_control(self, reference_image: Image.Image, 
                                               frame: Image.Image, 
                                               negative_samples: List[Image.Image] = None) -> float:
        """
        Compute similarity with negative control samples to improve discrimination.
        """
        # Preprocess images
        ref_img = self.preprocess_image(reference_image)
        frame_img = self.preprocess_image(frame)
        
        # Process images with CLIP processor
        images_to_process = [ref_img, frame_img]
        if negative_samples:
            # Process negative samples
            processed_negatives = [self.preprocess_image(neg) for neg in negative_samples]
            images_to_process.extend(processed_negatives)
        
        inputs = self.processor(
            images=images_to_process,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        
        # Get image features
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            # Normalize features
            features = torch.nn.functional.normalize(features, dim=-1)
            
        # Compute cosine similarity between reference and frame
        ref_to_frame_sim = torch.cosine_similarity(features[0].unsqueeze(0), features[1].unsqueeze(0))
        
        # If we have negative samples, compute similarity to them and subtract
        if negative_samples and len(negative_samples) > 0:
            # Compute average similarity to negative samples
            neg_similarities = []
            for i in range(2, 2 + len(negative_samples)):
                neg_sim = torch.cosine_similarity(features[0].unsqueeze(0), features[i].unsqueeze(0))
                neg_similarities.append(neg_sim)
            
            avg_neg_sim = torch.mean(torch.stack(neg_similarities))
            # Return differential similarity (reference similarity minus negative similarity)
            differential_sim = ref_to_frame_sim - avg_neg_sim
            return float(differential_sim.item())
        
        return float(ref_to_frame_sim.item())
    
    def detect_reference_in_video(self, reference_path: str, video_path: str, 
                                frame_interval: float = 0.5,
                                negative_sample_paths: List[str] = None) -> Dict:
        """
        Enhanced detection with negative sample comparison.
        """
        # Load reference image
        reference_image = Image.open(reference_path)
        
        # Load negative samples if provided
        negative_samples = []
        if negative_sample_paths:
            for neg_path in negative_sample_paths:
                if os.path.exists(neg_path):
                    negative_samples.append(Image.open(neg_path))
        
        # Extract frames
        frames = self.extract_frames(video_path, frame_interval)
        
        # Store results
        similarities = []
        timestamps = []
        
        print(f"Processing {len(frames)} frames...")
        
        # Compare each frame with reference image
        for i, (frame, timestamp) in enumerate(frames):
            similarity = self.compute_similarity_with_negative_control(
                reference_image, frame, negative_samples
            )
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
        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        
        # Enhanced detection algorithm:
        # 1. Require a minimum threshold
        # 2. Look for significant peaks above background
        # 3. Consider statistical significance
        
        min_threshold = 0.25  # Lowered threshold since we're using differential similarity
        
        # If max similarity is below our minimum threshold, it's definitely not found
        if max_similarity < min_threshold:
            return {
                "found": False,
                "confidence": 0.0,
                "max_similarity": max_similarity,
                "mean_similarity": mean_sim,
                "std_similarity": std_sim,
                "matches": [],
                "total_frames_processed": len(frames)
            }
        
        # Statistical approach - check if max is significantly above mean
        # Using z-score approach: (max - mean) / std
        if std_sim > 0:
            z_score = (max_similarity - mean_sim) / std_sim
        else:
            z_score = 0 if max_similarity == mean_sim else float('inf')
        
        # Require a significant statistical difference (z-score > 3)
        significant_peak = z_score > 3.0
        
        # Additional check: require a cluster of high similarity frames
        high_frames = similarities > max(0.2, mean_sim + std_sim)
        consecutive_high = 0
        max_consecutive = 0
        
        for is_high in high_frames:
            if is_high:
                consecutive_high += 1
                max_consecutive = max(max_consecutive, consecutive_high)
            else:
                consecutive_high = 0
        
        # Require at least 2 consecutive high frames
        has_cluster = max_consecutive >= 2
        
        # Final determination
        found = max_similarity >= min_threshold and significant_peak and has_cluster
        confidence = max_similarity if found else 0.0
        
        # Find detailed match information if found
        matches = []
        if found:
            # Use adaptive threshold based on statistics
            match_threshold = max(0.2, mean_sim + 0.5 * std_sim)
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
            "z_score": z_score,
            "matches": matches,
            "total_frames_processed": len(frames),
            "stats": {
                "significant_peak": significant_peak,
                "has_cluster": has_cluster
            }
        }

def main():
    """Main function for testing the enhanced analyzer."""
    # Initialize analyzer
    analyzer = EnhancedFrameAnalyzer()
    
    # Define paths (these would be parameters in a real implementation)
    base_path = "examples/thinktank"
    reference_path = os.path.join(base_path, "ReferenceImage.png")
    
    # Test with the files you mentioned
    test_files = []
    for file in os.listdir(base_path):
        if file.endswith('.mp4'):
            test_files.append(os.path.join(base_path, file))
    
    print("\n" + "="*70)
    print("ENHANCED FRAME FINDER - TESTING WITH EXAMPLE FILES")
    print("="*70)
    
    for video_path in test_files:
        print(f"\nTesting with video: {os.path.basename(video_path)}")
        print("-" * 50)
        
        if os.path.exists(video_path):
            result = analyzer.detect_reference_in_video(
                reference_path, 
                video_path, 
                frame_interval=0.5
            )
            
            print(f"Reference found: {result['found']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Max similarity: {result['max_similarity']:.3f}")
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
                print("Match timestamps (s):", [f"{m['timestamp']:.1f}" for m in result['matches'][:5]])
        else:
            print("Video file not found!")

if __name__ == "__main__":
    main()