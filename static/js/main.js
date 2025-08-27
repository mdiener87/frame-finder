// Frame Finder JavaScript functionality

// Update confidence threshold display
function updateThresholdValue() {
    const thresholdInput = document.getElementById('confidenceThreshold');
    const thresholdValue = document.getElementById('thresholdValue');
    
    if (thresholdInput && thresholdValue) {
        thresholdValue.textContent = thresholdInput.value;
    }
}

// Preview reference images
function previewReferenceImages() {
    const referenceInput = document.getElementById('reference_images');
    const preview = document.getElementById('referencePreview');
    
    if (referenceInput && preview) {
        preview.innerHTML = '';
        
        for (let i = 0; i < referenceInput.files.length; i++) {
            const file = referenceInput.files[i];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.className = 'thumbnail-preview img-thumbnail me-2 mb-2';
                    img.alt = file.name;
                    preview.appendChild(img);
                };
                reader.readAsDataURL(file);
            }
        }
    }
}

// List selected videos
function listSelectedVideos() {
    const videoInput = document.getElementById('videos');
    const directoryInput = document.getElementById('videoDirectory');
    const list = document.getElementById('videoList');
    
    if (list) {
        list.innerHTML = '<h6>Selected Videos:</h6><ul class="list-group">';
        
        // Handle individual file selection
        if (videoInput && videoInput.files.length > 0) {
            for (let i = 0; i < videoInput.files.length; i++) {
                const file = videoInput.files[i];
                const fileSizeMB = (file.size / (1024*1024)).toFixed(2);
                
                list.innerHTML += `<li class="list-group-item d-flex justify-content-between align-items-center">
                    ${file.name}
                    <span class="badge bg-primary rounded-pill">${fileSizeMB} MB</span>
                </li>`;
            }
        }
        
        // Handle directory selection
        if (directoryInput && directoryInput.files.length > 0) {
            for (let i = 0; i < directoryInput.files.length; i++) {
                const file = directoryInput.files[i];
                const fileSizeMB = (file.size / (1024*1024)).toFixed(2);
                
                list.innerHTML += `<li class="list-group-item d-flex justify-content-between align-items-center">
                    ${file.webkitRelativePath || file.name}
                    <span class="badge bg-primary rounded-pill">${fileSizeMB} MB</span>
                </li>`;
            }
        }
        
        list.innerHTML += '</ul>';
    }
}

// Preview negative reference images
function previewNegativeReferences() {
    const negativeInput = document.getElementById('negative_references');
    const preview = document.getElementById('negativePreview');
    
    if (negativeInput && preview) {
        preview.innerHTML = '';
        
        for (let i = 0; i < negativeInput.files.length; i++) {
            const file = negativeInput.files[i];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.className = 'thumbnail-preview img-thumbnail me-2 mb-2';
                    img.alt = file.name;
                    preview.appendChild(img);
                };
                reader.readAsDataURL(file);
            }
        }
    }
}

// Handle form submission
function handleFormSubmission() {
    const form = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (form && analyzeBtn) {
        form.addEventListener('submit', function(e) {
            e.preventDefault(); // Prevent default form submission
            
            analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            analyzeBtn.disabled = true;
            
            // Show progress container
            const progressContainer = document.getElementById('progressContainer');
            if (progressContainer) {
                progressContainer.style.display = 'block';
            }
            
            // Submit form via AJAX
            const formData = new FormData(form);
            
            // Handle directory uploads properly
            const directoryInput = document.getElementById('videoDirectory');
            if (directoryInput && directoryInput.files.length > 0) {
                // For directory uploads, we need to append all files individually
                for (let i = 0; i < directoryInput.files.length; i++) {
                    formData.append('videoDirectory', directoryInput.files[i]);
                }
            }
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.task_id) {
                    // Start polling for task status
                    pollTaskStatus(data.task_id);
                } else {
                    throw new Error('No task ID received');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error starting processing: ' + error.message);
                analyzeBtn.innerHTML = 'Analyze Videos';
                analyzeBtn.disabled = false;
                if (progressContainer) {
                    progressContainer.style.display = 'none';
                }
            });
        });
    }
}

// Poll for task status
function pollTaskStatus(taskId) {
    const interval = setInterval(() => {
        fetch(`/task_status/${taskId}`)
        .then(response => response.json())
        .then(data => {
            // Update progress UI
            const progressBar = document.getElementById('progressBar');
            const progressPercent = document.getElementById('progressPercent');
            const matchesFound = document.getElementById('matchesFound');
            
            if (progressBar && progressPercent) {
                progressBar.style.width = data.progress + '%';
                progressPercent.textContent = data.progress;
            }
            
            // Handle different task statuses
            if (data.status === 'completed') {
                clearInterval(interval);
                // Redirect to results page
                window.location.href = `/results/${taskId}`;
            } else if (data.status === 'error') {
                clearInterval(interval);
                alert('Error during processing: ' + data.error);
                const analyzeBtn = document.getElementById('analyzeBtn');
                if (analyzeBtn) {
                    analyzeBtn.innerHTML = 'Analyze Videos';
                    analyzeBtn.disabled = false;
                }
                const progressContainer = document.getElementById('progressContainer');
                if (progressContainer) {
                    progressContainer.style.display = 'none';
                }
            }
            // For 'processing' status, continue polling
        })
        .catch(error => {
            console.error('Error polling task status:', error);
        });
    }, 1000); // Poll every second
}

// Export results functionality
function exportResults() {
    // In a full implementation, this would export the results to a file
    alert('In a full implementation, this would export results to a file.');
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    const thresholdInput = document.getElementById('confidenceThreshold');
    if (thresholdInput) {
        thresholdInput.addEventListener('input', updateThresholdValue);
    }

    const referenceInput = document.getElementById('reference_images');
    if (referenceInput) {
        referenceInput.addEventListener('change', previewReferenceImages);
    }

    const negativeInput = document.getElementById('negative_references');
    if (negativeInput) {
        negativeInput.addEventListener('change', previewNegativeReferences);
    }

    const videoInput = document.getElementById('videos');
    if (videoInput) {
        videoInput.addEventListener('change', listSelectedVideos);
const directoryInput = document.getElementById('videoDirectory');
    if (directoryInput) {
}
        directoryInput.addEventListener('change', listSelectedVideos);
    }

    handleFormSubmission();
});