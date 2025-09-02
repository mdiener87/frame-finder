# validate_test_files.py
import os
import sys
import json
from analyzer import process_videos

def validate_test_files():
    """Validate the analysis functionality with the provided test files"""
    
    # Define paths to test files
    test_dir = "tests"
    reference_image = os.path.join(test_dir, "reference_image.png")
    test_video = os.path.join(test_dir, "TT_Start.mp4")
    
    # Check if files exist
    if not os.path.exists(reference_image):
        print(f"❌ Reference image not found: {reference_image}")
        return False
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return False
    
    print(f"✅ Found reference image: {reference_image}")
    print(f"✅ Found test video: {test_video}")
    
    # Get file information
    ref_size = os.path.getsize(reference_image)
    video_size = os.path.getsize(test_video)
    
    print(f"📄 Reference image size: {ref_size:,} bytes")
    print(f"🎬 Test video size: {video_size:,} bytes")
    
    # Test the analyzer with these specific files
    print("\n🔍 Starting analysis with test files...")
    
    try:
        # Process videos with reasonable parameters for testing
        results = process_videos(
            reference_paths=[reference_image],
            video_paths=[test_video],
            frame_interval=1.0,  # Process every 1 second
            lpips_threshold=0.7,  # Reasonable threshold
            clip_threshold=0.2,   # Reasonable threshold
            nms_iou_threshold=0.5,
            debounce_n=2,
            debounce_m=8
        )
        
        print("✅ Analysis completed successfully!")
        print(f"📊 Results structure: {type(results)}")
        
        # Analyze results structure
        if isinstance(results, dict):
            print(f"📋 Number of videos processed: {len(results)}")
            
            for video_name, video_data in results.items():
                print(f"\n📹 Video: {video_name}")
                
                if isinstance(video_data, dict):
                    print(f"   Keys: {list(video_data.keys())}")
                    
                    if 'matches' in video_data:
                        matches = video_data['matches']
                        print(f"   🔍 Number of matches found: {len(matches)}")
                        
                        # Show details of first few matches
                        for i, match in enumerate(matches[:5]):  # Show first 5 matches
                            print(f"     Match {i+1}:")
                            for key, value in match.items():
                                if key in ['timestamp', 'confidence', 'lpips_score', 'clip_score']:
                                    print(f"       {key}: {value:.4f}")
                                else:
                                    print(f"       {key}: {value}")
                    
                    if 'max_confidence' in video_data:
                        print(f"   📈 Max confidence: {video_data['max_confidence']:.4f}")
                        
                        # Check if we have any matches above reasonable thresholds
                        if 'matches' in video_data:
                            matches_above_threshold = [
                                m for m in video_data['matches'] 
                                if m.get('confidence', 0) > 0.5
                            ]
                            print(f"   🎯 Matches above 50% confidence: {len(matches_above_threshold)}")
                            
                            matches_above_higher_threshold = [
                                m for m in video_data['matches'] 
                                if m.get('confidence', 0) > 0.7
                            ]
                            print(f"   🔥 Matches above 70% confidence: {len(matches_above_higher_threshold)}")
                else:
                    print(f"   Unexpected video data type: {type(video_data)}")
                    print(f"   Video data: {video_data}")
        else:
            print(f"❌ Unexpected results type: {type(results)}")
            print(f"Results: {results}")
            
        # Save detailed results to a file for inspection
        results_file = os.path.join(test_dir, "analysis_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_file_properties():
    """Check properties of the test files"""
    import cv2
    from PIL import Image
    
    print("🔍 Checking file properties...")
    
    # Check reference image
    reference_image = os.path.join("tests", "reference_image.png")
    if os.path.exists(reference_image):
        try:
            # Using PIL
            img_pil = Image.open(reference_image)
            print(f"🖼️  Reference image (PIL): {img_pil.size} pixels, mode: {img_pil.mode}")
            
            # Using OpenCV
            img_cv = cv2.imread(reference_image)
            if img_cv is not None:
                print(f"📷 Reference image (OpenCV): {img_cv.shape[1]}x{img_cv.shape[0]} pixels, channels: {img_cv.shape[2] if len(img_cv.shape) > 2 else 1}")
            else:
                print("❌ Could not load reference image with OpenCV")
        except Exception as e:
            print(f"❌ Error checking reference image: {e}")
    
    # Check video
    test_video = os.path.join("tests", "TT_Start.mp4")
    if os.path.exists(test_video):
        try:
            cap = cv2.VideoCapture(test_video)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                
                print(f"🎬 Test video: {width}x{height} pixels, {fps} FPS, {frame_count} frames, {duration:.2f} seconds")
                cap.release()
            else:
                print("❌ Could not open test video")
        except Exception as e:
            print(f"❌ Error checking test video: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("VALIDATION TEST FOR PROVIDED TEST FILES")
    print("=" * 60)
    
    # Check file properties first
    check_file_properties()
    
    print("\n" + "=" * 60)
    print("RUNNING ANALYSIS VALIDATION")
    print("=" * 60)
    
    # Run the validation
    success = validate_test_files()
    
    if success:
        print("\n✅ Validation test completed successfully!")
        print("The analyzer is working with your test files.")
        print("If you're still not seeing results in the UI, the issue is likely in the frontend filtering or display.")
    else:
        print("\n❌ Validation test failed!")
        print("There's an issue with the analyzer processing your test files.")
        
    print("\n" + "=" * 60)