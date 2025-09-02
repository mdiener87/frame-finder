#!/usr/bin/env python3
"""
Unified API for the Frame Analyzer.
Supports analysis of single or multiple videos with a clean results interface.
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for
from frame_analyzer import FrameAnalyzer
import threading
import uuid
from typing import Dict, List
import json
import tempfile
import shutil
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size

# Initialize the frame analyzer
analyzer = FrameAnalyzer()

# In-memory storage for analysis results (in production, use a database)
analysis_results: Dict[str, Dict] = {}

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/analyze')
def analyze_page():
    """Serve the analysis page."""
    return render_template('analyze.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_videos():
    """Analyze one or more videos to determine which contain the reference image.
    
    Expected form data:
    - reference: reference image file
    - videos: one or more video files
    
    Returns:
    {
        "analysis_id": "unique-id",
        "status": "processing",
        "message": "Analysis started"
    }
    """
    try:
        print("Starting analysis request...")
        
        # Check if we have files
        if 'reference' not in request.files:
            return jsonify({
                'error': 'Missing reference image'
            }), 400
        
        # Check file count
        video_files = request.files.getlist('videos')
        if len(video_files) > 50:
            return jsonify({
                'error': 'Too many files selected. Please select 50 or fewer videos at a time.'
            }), 413  # Request Entity Too Large
        
        # Create temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")
        
        # Save reference image
        reference_file = request.files['reference']
        reference_path = os.path.join(temp_dir, reference_file.filename)
        reference_file.save(reference_path)
        print(f"Saved reference image: {reference_path}")
        
        # Save video files
        video_paths = []
        
        # Handle multiple files with the same key
        if 'videos' in request.files:
            video_files = request.files.getlist('videos')
            for i, video_file in enumerate(video_files):
                # Check individual file size (1GB limit per file)
                if len(video_files) > 1 and i == 0:  # Only check on first iteration to avoid multiple seeks
                    video_file.seek(0, os.SEEK_END)
                    file_size = video_file.tell()
                    video_file.seek(0)  # Reset file pointer
                    if file_size > 1024 * 1024 * 1024:  # 1GB
                        # Clean up and return error
                        shutil.rmtree(temp_dir)
                        return jsonify({
                            'error': f'File {video_file.filename} is too large (max 1GB per file)'
                        }), 413
                
                video_path = os.path.join(temp_dir, video_file.filename)
                video_file.save(video_path)
                video_paths.append(video_path)
                print(f"Saved video file {i+1}/{len(video_files)}: {video_path}")
        
        if not video_paths:
            # Clean up and return error
            shutil.rmtree(temp_dir)
            return jsonify({
                'error': 'No video files provided'
            }), 400
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        print(f"Generated analysis ID: {analysis_id}")
        
        # Store analysis information
        analysis_results[analysis_id] = {
            "status": "processing",
            "reference_path": reference_path,
            "video_paths": video_paths,
            "temp_dir": temp_dir,
            "results": {},
            "comparison": {}
        }
        
        # Start analysis in background thread
        thread = threading.Thread(
            target=perform_analysis,
            args=(analysis_id, reference_path, video_paths, temp_dir)
        )
        thread.start()
        print(f"Started analysis thread for {analysis_id}")
        
        return jsonify({
            "analysis_id": analysis_id,
            "status": "processing",
            "message": "Analysis started"
        })
        
    except Exception as e:
        print(f"Error in analyze_videos: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

def perform_analysis(analysis_id: str, reference_path: str, 
                    video_paths: List[str], temp_dir: str):
    """Perform analysis in background."""
    try:
        results = {}
        print(f"Starting analysis for {len(video_paths)} videos")
        
        # Analyze each video
        for i, video_path in enumerate(video_paths):
            try:
                print(f"Analyzing video {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")
                
                # Update status
                analysis_results[analysis_id]["current_video"] = os.path.basename(video_path)
                analysis_results[analysis_id]["progress"] = f"{i+1}/{len(video_paths)}"
                
                # Run detection
                result = analyzer.detect_reference_in_video(
                    reference_path, 
                    video_path, 
                    frame_interval=1.0  # Reasonable default
                )
                
                # Add video name to result for easier identification
                result["video_name"] = os.path.basename(video_path)
                results[video_path] = result
                print(f"Completed analysis for {os.path.basename(video_path)}")
                
            except Exception as e:
                print(f"Error analyzing video {video_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                results[video_path] = {
                    "error": str(e),
                    "found": False,
                    "confidence": 0.0,
                    "max_similarity": 0.0,
                    "matches": [],
                    "video_name": os.path.basename(video_path)
                }
        
        # Store raw results
        analysis_results[analysis_id]["results"] = results
        print(f"Stored raw results for {analysis_id}")
        
        # Perform comparative analysis
        comparison = perform_comparative_analysis(results)
        analysis_results[analysis_id]["comparison"] = comparison
        print(f"Performed comparative analysis for {analysis_id}")
        
        # Update status
        analysis_results[analysis_id]["status"] = "completed"
        print(f"Analysis {analysis_id} completed successfully")
        
    except Exception as e:
        print(f"Error in perform_analysis for {analysis_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        analysis_results[analysis_id]["status"] = "failed"
        analysis_results[analysis_id]["error"] = str(e)

def json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serializable(item) for item in obj]
    else:
        return obj

def perform_comparative_analysis(results: Dict) -> Dict:
    """Perform comparative analysis to determine which videos are most likely to contain the reference."""
    if not results:
        return {}
    
    # Extract metrics for each video
    video_metrics = {}
    for video_path, result in results.items():
        video_name = result.get("video_name", os.path.basename(video_path))
        if "error" in result:
            video_metrics[video_name] = {
                "max_similarity": 0.0,
                "mean_similarity": 0.0,
                "matches_count": 0,
                "confidence": 0.0,
                "found": False
            }
        else:
            video_metrics[video_name] = {
                "max_similarity": float(result.get("max_similarity", 0.0)),
                "mean_similarity": float(result.get("mean_similarity", 0.0)),
                "matches_count": len(result.get("matches", [])),
                "confidence": float(result.get("confidence", 0.0)),
                "found": bool(result.get("found", False))
            }
    
    # Calculate comparative scores
    if len(video_metrics) < 2:
        # If only one video, use absolute thresholds
        comparative_results = {}
        for video_name, metrics in video_metrics.items():
            # High confidence threshold approach
            is_likely = (
                metrics["max_similarity"] > 0.80 or  # High similarity
                (metrics["matches_count"] > 0 and metrics["max_similarity"] > 0.70)  # Some matches with decent similarity
            )
            
            comparative_results[video_name] = {
                "is_likely_match": bool(is_likely),
                "confidence_level": "high" if metrics["max_similarity"] > 0.85 else 
                                  "medium" if metrics["max_similarity"] > 0.75 else 
                                  "low",
                "max_similarity": float(metrics["max_similarity"]),
                "matches_count": int(metrics["matches_count"]),
                "reasoning": f"Max similarity: {metrics['max_similarity']:.3f}, Matches: {metrics['matches_count']}"
            }
    else:
        # Comparative analysis with multiple videos
        # Calculate differences
        max_matches = max([m["matches_count"] for m in video_metrics.values()])
        max_similarity = max([m["max_similarity"] for m in video_metrics.values()])
        
        comparative_results = {}
        for video_name, metrics in video_metrics.items():
            # Determine if this video stands out
            matches_ratio = metrics["matches_count"] / max(1, max_matches) if max_matches > 0 else 0
            similarity_ratio = metrics["max_similarity"] / max(1, max_similarity) if max_similarity > 0 else 0
            
            # Criteria for being "likely"
            has_significant_matches = matches_ratio > 0.5  # At least half the best match count
            has_high_similarity = similarity_ratio > 0.9  # Within 10% of the best similarity
            absolute_high_confidence = metrics["max_similarity"] > 0.80
            
            is_likely = (
                (has_significant_matches and has_high_similarity) or
                absolute_high_confidence or
                (metrics["matches_count"] > 0 and metrics["max_similarity"] > 0.75)
            )
            
            # Confidence level
            if metrics["max_similarity"] > 0.85:
                confidence_level = "high"
            elif metrics["max_similarity"] > 0.75:
                confidence_level = "medium"
            else:
                confidence_level = "low"
            
            # Reasoning
            if has_significant_matches and has_high_similarity:
                reasoning = f"Strong match: {metrics['matches_count']} matches, {metrics['max_similarity']:.3f} similarity"
            elif absolute_high_confidence:
                reasoning = f"High confidence: {metrics['max_similarity']:.3f} similarity"
            elif metrics["matches_count"] > 0:
                reasoning = f"Some matches: {metrics['matches_count']} matches, {metrics['max_similarity']:.3f} similarity"
            else:
                reasoning = f"Low confidence: {metrics['max_similarity']:.3f} similarity"
            
            comparative_results[video_name] = {
                "is_likely_match": bool(is_likely),
                "confidence_level": confidence_level,
                "max_similarity": float(metrics["max_similarity"]),
                "matches_count": int(metrics["matches_count"]),
                "matches_ratio": float(matches_ratio),
                "similarity_ratio": float(similarity_ratio),
                "reasoning": reasoning
            }
    
    return json_serializable(comparative_results)

@app.route('/api/analysis/<analysis_id>')
def get_analysis_results(analysis_id: str):
    """Get the results of an analysis."""
    try:
        if analysis_id not in analysis_results:
            return jsonify({
                'error': 'Analysis not found'
            }), 404
        
        result = analysis_results[analysis_id].copy()
        
        # Add current status if still processing
        if result["status"] == "processing":
            result["current_status"] = {
                "current_video": result.get("current_video", "Unknown"),
                "progress": result.get("progress", "0/0")
            }
        
        # Remove temp_dir from response for security
        if "temp_dir" in result:
            del result["temp_dir"]
        
        # Ensure all values are JSON serializable
        result = json_serializable(result)
        
        print(f"Returning results for analysis {analysis_id}: {result['status']}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in get_analysis_results for {analysis_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)