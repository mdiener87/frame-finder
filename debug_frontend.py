# debug_frontend.py
import os
import json
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Custom filter to extract basename from path
@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path) if path else ''

@app.route('/')
def debug_results():
    """Debug route to show exactly what the template receives"""
    
    # Simulate the exact data structure that would be passed to the template
    test_results = {
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
                },
                {
                    "bbox": [90, 44, 134, 134],
                    "score": 0.3267,
                    "scale": 0.6,
                    "reference_index": 0,
                    "timestamp": 3.003,
                    "lpips_score": 0.5771,
                    "clip_score": 0.5077,
                    "confidence": 0.6096,
                    "reference_image": "reference_image.png"
                }
            ],
            "max_confidence": 0.6793,
            "thresholds_used": {
                "lpips": 0.7,
                "clip": 0.2
            }
        }
    }
    
    # Save to a file for inspection
    debug_file = os.path.join("tests", "debug_template_data.json")
    os.makedirs("tests", exist_ok=True)
    
    with open(debug_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"Debug data saved to: {debug_file}")
    print("Template will receive this data structure:")
    print(json.dumps(test_results, indent=2, default=str))
    
    # Render the template with this data
    return render_template('results.html', results=test_results, task_id='debug-task-id')

@app.route('/raw-data')
def raw_data():
    """Return raw JSON data for debugging"""
    test_results = {
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
                }
            ],
            "max_confidence": 0.6793
        }
    }
    return jsonify(test_results)

if __name__ == '__main__':
    app.run(debug=True, port=5002)