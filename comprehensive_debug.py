# comprehensive_debug.py
import os
import json
import tempfile
from analyzer import process_videos
from flask import Flask, render_template

app = Flask(__name__)

# Custom filter to extract basename from path
@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path) if path else ''

def debug_full_pipeline():
    """Comprehensive debug of the entire pipeline"""
    
    print("=" * 60)
    print("COMPREHENSIVE DEBUG OF FRAME FINDER PIPELINE")
    print("=" * 60)
    
    # Step 1: Check if test files exist
    print("\n🔍 STEP 1: Checking test files...")
    reference_image = os.path.join("tests", "reference_image.png")
    test_video = os.path.join("tests", "TT_Start.mp4")
    
    if not os.path.exists(reference_image):
        print(f"❌ Reference image not found: {reference_image}")
        return False
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return False
    
    print(f"✅ Reference image: {reference_image}")
    print(f"✅ Test video: {test_video}")
    
    # Step 2: Run analyzer
    print("\n🔍 STEP 2: Running analyzer...")
    try:
        results = process_videos(
            reference_paths=[reference_image],
            video_paths=[test_video],
            frame_interval=1.0,
            lpips_threshold=0.7,
            clip_threshold=0.2
        )
        print("✅ Analyzer completed successfully!")
    except Exception as e:
        print(f"❌ Analyzer failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Inspect results structure
    print("\n🔍 STEP 3: Inspecting results structure...")
    print(f"Results type: {type(results)}")
    
    if not isinstance(results, dict):
        print("❌ Results is not a dictionary!")
        print(f"Results: {results}")
        return False
    
    print(f"Number of videos processed: {len(results)}")
    
    for video_name, video_data in results.items():
        print(f"\n📹 Video: {video_name}")
        print(f"  Data type: {type(video_data)}")
        
        if not isinstance(video_data, dict):
            print(f"  ❌ Video data is not a dictionary!")
            print(f"  Video data: {video_data}")
            continue
            
        print(f"  Keys: {list(video_data.keys())}")
        
        if 'matches' not in video_data:
            print(f"  ❌ No 'matches' key found!")
            continue
            
        matches = video_data['matches']
        print(f"  🔍 Number of matches: {len(matches)}")
        
        if len(matches) > 0:
            print(f"  📊 First match details:")
            first_match = matches[0]
            for key, value in first_match.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
        
        if 'max_confidence' in video_data:
            print(f"  📈 Max confidence: {video_data['max_confidence']:.4f}")
    
    # Step 4: Save debug data for template inspection
    print("\n🔍 STEP 4: Saving debug data...")
    debug_file = os.path.join("tests", "debug_results.json")
    with open(debug_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Debug data saved to: {debug_file}")
    
    # Step 5: Test template rendering
    print("\n🔍 STEP 5: Testing template rendering...")
    
    # Create a minimal Flask app to test template rendering
    with app.test_request_context():
        try:
            # Render template with the results
            rendered = render_template('results.html', results=results, task_id='debug-task-id')
            print("✅ Template rendered successfully!")
            
            # Check if the rendered content contains expected elements
            if 'TT_Start.mp4' in rendered:
                print("✅ Video name found in rendered template")
            else:
                print("❌ Video name NOT found in rendered template")
                
            if str(len(matches)) in rendered:
                print("✅ Match count found in rendered template")
            else:
                print("❌ Match count NOT found in rendered template")
                
            # Save rendered template for inspection
            rendered_file = os.path.join("tests", "rendered_template.html")
            with open(rendered_file, 'w') as f:
                f.write(rendered)
            print(f"✅ Rendered template saved to: {rendered_file}")
            
        except Exception as e:
            print(f"❌ Template rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Step 6: Create a standalone HTML file for manual testing
    print("\n🔍 STEP 6: Creating standalone HTML test file...")
    create_standalone_test_html(results)
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
    
    return True

def create_standalone_test_html(results):
    """Create a standalone HTML file to test the exact data"""
    
    # Create HTML content with the exact data
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Debug Test - Frame Finder Results</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <div class="row">
            <div class="col-md-12">
                <h1>Debug Test - Frame Finder Results</h1>
                <p class="lead">Testing with actual analyzer results</p>
                
                <!-- Global confidence filter -->
                <div class="card mb-4">
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h5>Global Confidence Filter</h5>
                            </div>
                            <div class="col-md-6">
                                <div class="mb-2">
                                    <label for="globalDynamicThreshold" class="form-label">Filter by Confidence:</label>
                                    <input type="range" class="form-range" id="globalDynamicThreshold" min="0" max="100" value="0">
                                    <div class="form-text">
                                        Minimum confidence: <span id="globalDynamicThresholdValue">0</span>%
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Video Results -->
                """
    
    # Add video results
    for video_index, (video_name, video_data) in enumerate(results.items(), 1):
        if isinstance(video_data, dict) and 'matches' in video_data:
            matches = video_data['matches']
            max_confidence = video_data.get('max_confidence', 0.0)
            
            html_content += f"""
                <div class="card result-card mb-4">
                    <div class="card-header">
                        <h3>{video_name}</h3>
                    </div>
                    <div class="card-body">
                        <div class="row mb-3">
                            <div class="col-md-12">
                                <p class="text-muted">
                                    <span id="visibleCount-{video_index}">{len(matches)}</span> of {len(matches)} match(es) found |
                                    Max confidence: <span class="max-confidence">{max_confidence * 100:.2f}%</span>
                                </p>
                            </div>
                        </div>
                        
                        <!-- Confidence Filters -->
                        <div class="card mb-4">
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-12">
                                        <h5>Confidence Filters</h5>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="mb-2">
                                            <label for="overallThreshold-{video_index}" class="form-label">Overall Confidence:</label>
                                            <input type="range" class="form-range overall-threshold" id="overallThreshold-{video_index}" min="0" max="100" value="0" data-video-index="{video_index}">
                                            <div class="form-text">
                                                Minimum: <span id="overallThresholdValue-{video_index}">0</span>%
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="mb-2">
                                            <label for="lpipsFilter-{video_index}" class="form-label">LPIPS Score:</label>
                                            <input type="range" class="form-range lpips-filter" id="lpipsFilter-{video_index}" min="0" max="100" value="0" data-video-index="{video_index}">
                                            <div class="form-text">
                                                Minimum: <span id="lpipsFilterValue-{video_index}">0</span>%
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="mb-2">
                                            <label for="clipFilter-{video_index}" class="form-label">CLIP Score:</label>
                                            <input type="range" class="form-range clip-filter" id="clipFilter-{video_index}" min="0" max="100" value="0" data-video-index="{video_index}">
                                            <div class="form-text">
                                                Minimum: <span id="clipFilterValue-{video_index}">0</span>%
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Results Table -->
                        <div class="table-responsive">
                            <table class="table table-striped results-table" id="resultsTable-{video_index}">
                                <thead>
                                    <tr>
                                        <th>Timestamp</th>
                                        <th>Overall Confidence</th>
                                        <th>LPIPS Score</th>
                                        <th>CLIP Score</th>
                                        <th>Reference Image</th>
                                    </tr>
                                </thead>
                                <tbody>
                                """
            
            # Add matches
            for match in matches[:10]:  # Show first 10 matches for testing
                confidence = match.get('confidence', 0.0)
                lpips_score = match.get('lpips_score', 0.0)
                clip_score = match.get('clip_score', 0.0)
                timestamp = match.get('timestamp', 0.0)
                reference_image = match.get('reference_image', '')
                
                # Determine row class based on confidence
                if confidence > 0.8:
                    row_class = 'table-success'
                elif confidence > 0.6:
                    row_class = 'table-warning'
                else:
                    row_class = 'table-secondary'
                
                html_content += f"""
                                    <tr class="{row_class}"
                                        data-confidence="{confidence}"
                                        data-lpips="{lpips_score}"
                                        data-clip="{clip_score}">
                                        <td class="timestamp" data-timestamp="{timestamp}">{timestamp:.3f}s</td>
                                        <td class="confidence">{confidence * 100:.2f}%</td>
                                        <td class="lpips-score">{lpips_score * 100:.2f}%</td>
                                        <td class="clip-score">{clip_score * 100:.2f}%</td>
                                        <td>
                                            <span class="badge bg-secondary">{os.path.basename(reference_image) if reference_image else ''}</span>
                                        </td>
                                    </tr>
                                """
            
            html_content += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                """
    
    # Add JavaScript
    html_content += """
            </div>
        </div>
    </div>

    <script>
        // Format seconds to hh:mm:ss format
        function formatTime(seconds) {
            const hrs = Math.floor(seconds / 3600);
            const mins = Math.floor((seconds % 3600) / 60);
            const secs = seconds % 60;
            return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toFixed(3).padStart(6, '0')}`;
        }
        
        // Update timestamp displays
        function updateTimestamps() {
            document.querySelectorAll('.timestamp[data-timestamp]').forEach(function(element) {
                const seconds = parseFloat(element.getAttribute('data-timestamp'));
                element.textContent = formatTime(seconds);
            });
        }
        
        // Initialize when DOM is loaded
        document.addEventListener('DOMContentLoaded', function() {
            // Update timestamps
            updateTimestamps();
            
            // Add event listener for global dynamic threshold slider
            const globalSlider = document.getElementById('globalDynamicThreshold');
            const globalValueDisplay = document.getElementById('globalDynamicThresholdValue');
            
            if (globalSlider && globalValueDisplay) {
                globalValueDisplay.textContent = globalSlider.value;
                
                globalSlider.addEventListener('input', function() {
                    const threshold = this.value;
                    globalValueDisplay.textContent = threshold;
                    
                    // Filter rows in all tables based on confidence threshold
                    document.querySelectorAll('.results-table').forEach(function(resultsTable) {
                        const videoIndex = resultsTable.id.replace('resultsTable-', '');
                        const visibleCount = document.getElementById('visibleCount-' + videoIndex);
                        
                        let visibleRowCount = 0;
                        resultsTable.querySelectorAll('tbody tr').forEach(function(row) {
                            const confidence = parseFloat(row.getAttribute('data-confidence'));
                            if (confidence >= threshold / 100.0) {
                                row.style.display = '';
                                visibleRowCount++;
                            } else {
                                row.style.display = 'none';
                            }
                        });
                        
                        if (visibleCount) {
                            visibleCount.textContent = visibleRowCount;
                        }
                    });
                });
                
                // Trigger initial filtering
                globalSlider.dispatchEvent(new Event('input'));
            }
        });
    </script>
</body>
</html>
"""
    
    # Save the standalone HTML file
    standalone_file = os.path.join("tests", "standalone_debug.html")
    with open(standalone_file, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Standalone HTML test file saved to: {standalone_file}")

if __name__ == "__main__":
    debug_full_pipeline()