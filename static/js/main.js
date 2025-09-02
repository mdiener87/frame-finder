// Frame Finder JavaScript functionality


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
            
            // Save settings before processing
            saveSettings();
            
            analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            analyzeBtn.disabled = true;
            
            // Disable all settings
            disableSettings(true);
            
            // Collapse settings panel and expand progress panel
            const settingsPanel = document.getElementById('settingsPanel');
            const progressPanel = document.getElementById('progressPanel');
            const cancelBtn = document.getElementById('cancelBtn');
            
            if (settingsPanel) {
                settingsPanel.classList.remove('show');
            }
            
            if (progressPanel) {
                progressPanel.classList.add('show');
            }
            
            // Show cancel button
            if (cancelBtn) {
                cancelBtn.style.display = 'inline-block';
            }
            
            // Submit form via AJAX
            const formData = new FormData(form);
            
            // Remove confidenceThreshold from formData since we removed the slider
            formData.delete('confidenceThreshold');
            
            // Handle reference images properly
            const referenceInput = document.getElementById('reference_images');
            if (referenceInput && referenceInput.files.length > 0) {
                // Clear the existing reference_images entries
                formData.delete('reference_images');
                // For multiple file uploads, we need to append all files individually
                for (let i = 0; i < referenceInput.files.length; i++) {
                    formData.append('reference_images', referenceInput.files[i]);
                }
            }
            
            // Handle negative reference images properly
            const negativeInput = document.getElementById('negative_references');
            if (negativeInput && negativeInput.files.length > 0) {
                // Clear the existing negative_references entries
                formData.delete('negative_references');
                // For multiple file uploads, we need to append all files individually
                for (let i = 0; i < negativeInput.files.length; i++) {
                    formData.append('negative_references', negativeInput.files[i]);
                }
            }
            
            // Handle video files properly
            const videoInput = document.getElementById('videos');
            if (videoInput && videoInput.files.length > 0) {
                // Clear the existing videos entries
                formData.delete('videos');
                // For multiple file uploads, we need to append all files individually
                for (let i = 0; i < videoInput.files.length; i++) {
                    formData.append('videos', videoInput.files[i]);
                }
            }
            
            // Handle directory uploads properly
            const directoryInput = document.getElementById('videoDirectory');
            if (directoryInput && directoryInput.files.length > 0) {
                // Clear the existing videoDirectory entries
                formData.delete('videoDirectory');
                // For directory uploads, we need to append all files individually
                for (let i = 0; i < directoryInput.files.length; i++) {
                    formData.append('videoDirectory', directoryInput.files[i]);
                }
            }
            
            // Add new parameters to formData
            const frameStride = document.getElementById('frameStride');
            const resolutionTarget = document.getElementById('resolutionTarget');
            const lpipsThreshold = document.getElementById('lpipsThreshold');
            const clipThreshold = document.getElementById('clipThreshold');
            const nmsThreshold = document.getElementById('nmsThreshold');
            const debounceN = document.getElementById('debounceN');
            const debounceM = document.getElementById('debounceM');
            
            if (frameStride) formData.append('frameStride', frameStride.value);
            if (resolutionTarget) formData.append('resolutionTarget', resolutionTarget.value);
            if (lpipsThreshold) formData.append('lpipsThreshold', lpipsThreshold.value);
            if (clipThreshold) formData.append('clipThreshold', clipThreshold.value);
            if (nmsThreshold) formData.append('nmsThreshold', nmsThreshold.value);
            if (debounceN) formData.append('debounceN', debounceN.value);
            if (debounceM) formData.append('debounceM', debounceM.value);
            
            fetch('/upload', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                // Check if response is ok (status in the range 200-299)
                if (!response.ok) {
                    // Try to parse error JSON
                    return response.text().then(text => {
                        try {
                            const errorData = JSON.parse(text);
                            // Extract the error message from the JSON
                            const errorMessage = errorData.error || 'Unknown error occurred';
                            throw new Error(errorMessage);
                        } catch (e) {
                            // If JSON parsing fails, throw a generic error with the text content
                            throw new Error(text || 'Server error: ' + response.status);
                        }
                    });
                }
                // Parse successful response as JSON
                return response.json();
            })
            .then(data => {
                if (data.task_id) {
                    // Store taskId for cancel button
                    window.currentTaskId = data.task_id;
                    // Start polling for task status
                    pollTaskStatus(data.task_id);
                } else {
                    throw new Error('No task ID received');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error starting processing: ' + error.message);
                resetUI();
            });
        });
    }
}

// Add event listeners for new sliders
document.addEventListener('DOMContentLoaded', function() {
    const lpipsThreshold = document.getElementById('lpipsThreshold');
    const clipThreshold = document.getElementById('clipThreshold');
    const nmsThreshold = document.getElementById('nmsThreshold');
    const debounceN = document.getElementById('debounceN');
    const debounceM = document.getElementById('debounceM');
    
    if (lpipsThreshold) {
        lpipsThreshold.addEventListener('input', function() {
            document.getElementById('lpipsThresholdValue').textContent = this.value;
        });
    }
    
    if (clipThreshold) {
        clipThreshold.addEventListener('input', function() {
            document.getElementById('clipThresholdValue').textContent = this.value;
        });
    }
    
    if (nmsThreshold) {
        nmsThreshold.addEventListener('input', function() {
            document.getElementById('nmsThresholdValue').textContent = this.value;
        });
    }
    
    if (debounceN) {
        debounceN.addEventListener('input', function() {
            document.getElementById('debounceNValue').textContent = this.value;
        });
    }
    
    if (debounceM) {
        debounceM.addEventListener('input', function() {
            document.getElementById('debounceMValue').textContent = this.value;
        });
    }
    
    // Add event listeners for preset buttons
    document.querySelectorAll('.preset-btn').forEach(button => {
        button.addEventListener('click', function() {
            applyPreset(this.dataset.preset);
        });
    });
});

// Preset application function
function applyPreset(preset) {
    switch(preset) {
        case 'precision':
            // Stricter thresholds for high precision
            document.getElementById('lpipsThreshold').value = 0.25;
            document.getElementById('clipThreshold').value = 0.38;
            document.getElementById('nmsThreshold').value = 0.5;
            document.getElementById('debounceN').value = 4;
            document.getElementById('debounceM').value = 12;
            break;
        case 'recall':
            // Looser thresholds for high recall
            document.getElementById('lpipsThreshold').value = 0.40;
            document.getElementById('clipThreshold').value = 0.28;
            document.getElementById('nmsThreshold').value = 0.5;
            document.getElementById('debounceN').value = 2;
            document.getElementById('debounceM').value = 10;
            break;
        case 'balanced':
            // Default balanced settings
            document.getElementById('lpipsThreshold').value = 0.35;
            document.getElementById('clipThreshold').value = 0.33;
            document.getElementById('nmsThreshold').value = 0.5;
            document.getElementById('debounceN').value = 3;
            document.getElementById('debounceM').value = 12;
            break;
        case 'performance':
            // Settings optimized for performance
            document.getElementById('frameStride').value = 3;
            document.getElementById('resolutionTarget').value = 720;
            document.getElementById('lpipsThreshold').value = 0.35;
            document.getElementById('clipThreshold').value = 0.33;
            document.getElementById('nmsThreshold').value = 0.5;
            document.getElementById('debounceN').value = 2;
            document.getElementById('debounceM').value = 8;
            break;
    }
    
    // Update displayed values
    document.getElementById('lpipsThresholdValue').textContent = document.getElementById('lpipsThreshold').value;
    document.getElementById('clipThresholdValue').textContent = document.getElementById('clipThreshold').value;
    document.getElementById('nmsThresholdValue').textContent = document.getElementById('nmsThreshold').value;
    document.getElementById('debounceNValue').textContent = document.getElementById('debounceN').value;
    document.getElementById('debounceMValue').textContent = document.getElementById('debounceM').value;
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
            const currentVideo = document.getElementById('currentVideo');
            
            if (progressBar && progressPercent) {
                progressBar.style.width = data.progress + '%';
                progressPercent.textContent = Math.round(data.progress);
            }
            
            // Update current video name
            if (currentVideo && data.current_video) {
                currentVideo.textContent = data.current_video;
            }
            
            // Handle different task statuses
            if (data.status === 'completed') {
                clearInterval(interval);
                // Redirect to results page
                window.location.href = `/results/${taskId}`;
            } else if (data.status === 'error') {
                clearInterval(interval);
                alert('Error during processing: ' + data.error);
                resetUI();
            } else if (data.status === 'cancelled') {
                clearInterval(interval);
                alert('Processing was cancelled.');
                resetUI();
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
    // Get task ID from the export button
    const exportBtn = document.getElementById('exportBtn');
    if (!exportBtn) {
        alert('Export button not found.');
        return;
    }
    
    const taskId = exportBtn.getAttribute('data-task-id');
    if (!taskId) {
        alert('Task ID not found.');
        return;
    }
    
    // Get global confidence threshold
    const globalSlider = document.getElementById('globalDynamicThreshold');
    const globalThreshold = globalSlider ? parseInt(globalSlider.value) : 75;
    
    // Create confidence thresholds object with global threshold for all videos
    const confidenceThresholds = {};
    // Get video names from the result cards
    document.querySelectorAll('.result-card').forEach(function(card) {
        const videoName = card.querySelector('h3').textContent;
        if (videoName) {
            confidenceThresholds[videoName] = globalThreshold;
        }
    });
    
    // Send export request to backend
    fetch(`/export_results/${taskId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            confidence_thresholds: confidenceThresholds
        })
    })
    .then(response => {
        if (response.ok) {
            return response.blob();
        } else {
            throw new Error('Export failed');
        }
    })
    .then(blob => {
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `frame_finder_export_${taskId}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    })
    .catch(error => {
        console.error('Export error:', error);
        alert('Export failed: ' + error.message);
    });
}

// Cancel task functionality
function cancelTask(taskId) {
    // Show confirmation dialog
    if (confirm('Are you sure you want to cancel the analysis?')) {
        // Send cancel request to backend
        fetch(`/cancel_task/${taskId}`, {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                alert('Analysis cancelled.');
                resetUI();
            } else {
                throw new Error(data.error || 'Failed to cancel task');
            }
        })
        .catch(error => {
            console.error('Error cancelling task:', error);
            alert('Error cancelling task: ' + error.message);
        });
    }
}

// Disable or enable all settings controls
function disableSettings(disable) {
    const form = document.getElementById('uploadForm');
    if (form) {
        const inputs = form.querySelectorAll('input, select, textarea, button');
        inputs.forEach(input => {
            if (input.id !== 'analyzeBtn') {  // Don't disable the analyze button
                input.disabled = disable;
            }
        });
    }
}

// Save form settings to localStorage
function saveSettings() {
    const frameInterval = document.getElementById('frameInterval');
    
    if (frameInterval) {
        const settings = {
            frameInterval: frameInterval.value,
            timestamp: Date.now()
        };
        
        try {
            localStorage.setItem('frameFinderSettings', JSON.stringify(settings));
        } catch (e) {
            console.warn('Could not save settings to localStorage:', e);
        }
    }
}

// Reset UI after analysis completes or is cancelled
function resetUI() {
    // Reset analyze button
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.innerHTML = 'Analyze Videos';
        analyzeBtn.disabled = false;
    }
    
    // Expand settings panel and collapse progress panel
    const settingsPanel = document.getElementById('settingsPanel');
    const progressPanel = document.getElementById('progressPanel');
    const cancelBtn = document.getElementById('cancelBtn');
    
    if (settingsPanel) {
        settingsPanel.classList.add('show');
    }
    
    if (progressPanel) {
        progressPanel.classList.remove('show');
    }
    
    // Hide cancel button
    if (cancelBtn) {
        cancelBtn.style.display = 'none';
    }
    
    // Enable settings
    disableSettings(false);
}

// Restore form settings from localStorage
function restoreSettings() {
    try {
        const settingsStr = localStorage.getItem('frameFinderSettings');
        if (settingsStr) {
            const settings = JSON.parse(settingsStr);
            
            // Check if settings are recent (less than 1 hour old)
            if (Date.now() - settings.timestamp < 3600000) {
                const frameInterval = document.getElementById('frameInterval');
                
                if (frameInterval) {
                    frameInterval.value = settings.frameInterval;
                }
            }
        }
    } catch (e) {
        console.warn('Could not restore settings from localStorage:', e);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Restore settings from localStorage
    restoreSettings();
    

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
    }
    const directoryInput = document.getElementById('videoDirectory');
    if (directoryInput) {
        directoryInput.addEventListener('change', listSelectedVideos);
    }
// Add event listener for cancel button
    const cancelBtn = document.getElementById('cancelBtn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', function() {
            if (window.currentTaskId) {
                cancelTask(window.currentTaskId);
            } else {
                alert('No task is currently running.');
            }
        });
    }

    handleFormSubmission();
});