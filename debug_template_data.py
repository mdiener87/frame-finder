# debug_template_data.py
import os
import json
import tempfile
from analyzer import process_videos

def debug_template_data():
    """Debug what data is being passed to the results template"""
    
    # Use the test files
    reference_image = os.path.join("tests", "reference_image.png")
    test_video = os.path.join("tests", "TT_Start.mp4")
    
    # Check if files exist
    if not os.path.exists(reference_image):
        print(f"❌ Reference image not found: {reference_image}")
        return
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return
    
    print(f"✅ Using reference image: {reference_image}")
    print(f"✅ Using test video: {test_video}")
    
    # Process videos with debug info
    print("\n🔍 Processing videos...")
    results = process_videos(
        reference_paths=[reference_image],
        video_paths=[test_video],
        frame_interval=1.0,
        lpips_threshold=0.7,
        clip_threshold=0.2
    )
    
    print("✅ Processing completed!")
    
    # Save the exact data structure being passed to the template
    debug_data = {
        "results": results,
        "task_id": "debug-task-id",
        "data_type": str(type(results)),
        "data_keys": list(results.keys()) if isinstance(results, dict) else "Not a dict"
    }
    
    # Save to file for inspection
    debug_file = os.path.join("tests", "template_debug_data.json")
    with open(debug_file, 'w') as f:
        json.dump(debug_data, f, indent=2, default=str)
    
    print(f"\n💾 Debug data saved to: {debug_file}")
    print(f"📊 Results type: {type(results)}")
    
    # Print detailed structure analysis
    if isinstance(results, dict):
        print(f"📋 Number of videos: {len(results)}")
        
        for video_name, video_data in results.items():
            print(f"\n📹 Video: {video_name}")
            print(f"   Data type: {type(video_data)}")
            
            if isinstance(video_data, dict):
                print(f"   Keys: {list(video_data.keys())}")
                
                if 'matches' in video_data:
                    matches = video_data['matches']
                    print(f"   🔍 Number of matches: {len(matches)}")
                    
                    # Show first few matches
                    for i, match in enumerate(matches[:3]):
                        print(f"     Match {i+1}:")
                        for key, value in match.items():
                            if isinstance(value, (int, float)):
                                print(f"       {key}: {value:.4f}")
                            else:
                                print(f"       {key}: {value}")
                
                if 'max_confidence' in video_data:
                    print(f"   📈 Max confidence: {video_data['max_confidence']:.4f}")
            else:
                print(f"   Unexpected data type: {type(video_data)}")
                print(f"   Data: {video_data}")
    
    # Also save the raw results to a separate file
    raw_results_file = os.path.join("tests", "raw_results.json")
    with open(raw_results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Raw results also saved to: {raw_results_file}")
    
    # Create a simplified version that matches what the template expects
    simplified_results = {}
    if isinstance(results, dict):
        for video_name, video_data in results.items():
            if isinstance(video_data, dict):
                simplified_results[video_name] = {
                    'matches': video_data.get('matches', []),
                    'max_confidence': video_data.get('max_confidence', 0.0)
                }
                # Add any other keys that might be needed
                for key, value in video_data.items():
                    if key not in ['matches', 'max_confidence']:
                        simplified_results[video_name][key] = value
    
    simplified_file = os.path.join("tests", "simplified_results.json")
    with open(simplified_file, 'w') as f:
        json.dump(simplified_results, f, indent=2, default=str)
    
    print(f"💾 Simplified results saved to: {simplified_file}")
    
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("DEBUG TEMPLATE DATA STRUCTURE")
    print("=" * 60)
    
    debug_template_data()
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)