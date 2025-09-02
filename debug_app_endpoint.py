# debug_app_endpoint.py
import os
import json
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Load the actual data from our test
def get_test_data():
    """Load the test data we know works"""
    test_data_file = os.path.join("tests", "simplified_results.json")
    if os.path.exists(test_data_file):
        with open(test_data_file, 'r') as f:
            data = json.load(f)
        return data
    else:
        # Return sample data if test data not found
        return {
            "TT_Start.mp4": {
                "matches": [
                    {
                        "bbox": [4, 7, 246, 246],
                        "score": 1.0,
                        "scale": 1.1,
                        "reference_index": 0,
                        "timestamp": 0.0,
                        "lpips_score": 0.5861,
                        "clip_score": 0.2171,
                        "confidence": 0.4620,
                        "reference_image": "reference_image.png"
                    },
                    {
                        "bbox": [90, 44, 134, 134],
                        "score": 0.3283,
                        "scale": 0.6,
                        "reference_index": 0,
                        "timestamp": 4.004,
                        "lpips_score": 0.5762,
                        "clip_score": 0.5112,
                        "confidence": 0.6116,
                        "reference_image": "reference_image.png"
                    }
                ],
                "max_confidence": 0.6793
            }
        }

@app.route('/debug-results')
def debug_results():
    """Debug endpoint that shows exactly what the template receives"""
    results = get_test_data()
    
    # Log what we're passing to the template
    print("=== DEBUG RESULTS ENDPOINT ===")
    print(f"Results type: {type(results)}")
    print(f"Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
    
    if isinstance(results, dict):
        for video_name, video_data in results.items():
            print(f"Video: {video_name}")
            print(f"  Data type: {type(video_data)}")
            if isinstance(video_data, dict) and 'matches' in video_data:
                matches = video_data['matches']
                print(f"  Number of matches: {len(matches)}")
                for i, match in enumerate(matches[:3]):  # Show first 3 matches
                    print(f"    Match {i+1}: confidence={match.get('confidence', 'N/A')}")
    
    return render_template('results_fixed.html', results=results, task_id='debug-task-id')

@app.route('/debug-json')
def debug_json():
    """Debug endpoint that returns raw JSON"""
    results = get_test_data()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5003)