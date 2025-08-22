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
    const list = document.getElementById('videoList');
    
    if (videoInput && list) {
        list.innerHTML = '<h6>Selected Videos:</h6><ul class="list-group">';
        
        for (let i = 0; i < videoInput.files.length; i++) {
            const file = videoInput.files[i];
            const fileSizeMB = (file.size / (1024*1024)).toFixed(2);
            
            list.innerHTML += `<li class="list-group-item d-flex justify-content-between align-items-center">
                ${file.name}
                <span class="badge bg-primary rounded-pill">${fileSizeMB} MB</span>
            </li>`;
        }
        
        list.innerHTML += '</ul>';
    }
}

// Handle form submission
function handleFormSubmission() {
    const form = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (form && analyzeBtn) {
        form.addEventListener('submit', function() {
            analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            analyzeBtn.disabled = true;
        });
    }
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
    
    const videoInput = document.getElementById('videos');
    if (videoInput) {
        videoInput.addEventListener('change', listSelectedVideos);
    }
    
    handleFormSubmission();
});