// Frame Finder JavaScript functionality


// Reference images as removable badges with hover preview
let removedReferenceKeys = new Set();
function previewReferenceImages() {
    const referenceInput = document.getElementById('reference_images');
    const container = document.getElementById('referencePreview');
    if (!container) return;
    container.innerHTML = '';
    const all = [];
    if (referenceInput && referenceInput.files) {
        for (let i = 0; i < referenceInput.files.length; i++) all.push(referenceInput.files[i]);
    }
    if (typeof droppedReferences !== 'undefined' && droppedReferences.length) {
        all.push(...droppedReferences);
    }
    const visible = all.filter(f => !removedReferenceKeys.has(fileKey(f)));
    if (visible.length === 0) return;

    const header = document.createElement('h6');
    header.className = 'mb-2';
    header.textContent = `Reference Images (${visible.length}):`;
    container.appendChild(header);

    const badges = document.createElement('div');
    badges.className = 'ref-badges';

    let refPreviewEl;
    function ensureRefPreview() {
        if (!refPreviewEl) {
            refPreviewEl = document.createElement('div');
            refPreviewEl.style.position = 'fixed';
            refPreviewEl.style.pointerEvents = 'none';
            refPreviewEl.style.border = '1px solid rgba(0,0,0,0.15)';
            refPreviewEl.style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';
            refPreviewEl.style.background = '#fff';
            refPreviewEl.style.padding = '4px';
            refPreviewEl.style.zIndex = '9999';
            refPreviewEl.style.display = 'none';
            const img = document.createElement('img');
            img.style.maxWidth = '320px';
            img.style.maxHeight = '240px';
            img.style.display = 'block';
            refPreviewEl.appendChild(img);
            document.body.appendChild(refPreviewEl);
        }
    }

    visible.forEach((file) => {
        const key = fileKey(file);
        const name = file.name;
        const sizeKB = (file.size / 1024).toFixed(0);
        const badge = document.createElement('span');
        badge.className = 'ref-badge';
        badge.title = name;
        badge.innerHTML = `
            <span class="name">${name}</span>
            <span class="size">- ${sizeKB} KB</span>
            <button type="button" class="remove" aria-label="Remove" title="Remove">×</button>
        `;
        // Remove
        badge.querySelector('.remove').addEventListener('click', function(){
            removedReferenceKeys.add(key);
            previewReferenceImages();
        });
        // Hover preview
        badge.addEventListener('mouseenter', function(e){
            ensureRefPreview();
            const reader = new FileReader();
            reader.onload = function(ev){
                refPreviewEl.firstChild.src = ev.target.result;
                refPreviewEl.style.display = 'block';
            };
            reader.readAsDataURL(file);
        });
        badge.addEventListener('mousemove', function(e){
            if (!refPreviewEl) return;
            const margin = 12;
            let x = e.clientX + margin;
            let y = e.clientY + margin;
            const rect = refPreviewEl.getBoundingClientRect();
            if (x + rect.width > window.innerWidth) x = e.clientX - rect.width - margin;
            if (y + rect.height > window.innerHeight) y = e.clientY - rect.height - margin;
            refPreviewEl.style.left = x + 'px';
            refPreviewEl.style.top = y + 'px';
        });
        badge.addEventListener('mouseleave', function(){
            if (refPreviewEl) refPreviewEl.style.display = 'none';
        });
        badges.appendChild(badge);
    });
    container.appendChild(badges);
}

// List selected videos
// Track removed videos (by unique key) to allow badge removal
let removedVideoKeys = new Set();
function fileKey(file) { return `${file.name}|${file.size}|${file.lastModified}`; }
function listSelectedVideos() {
    const videoInput = document.getElementById('videos');
    const directoryInput = document.getElementById('videoDirectory');
    const list = document.getElementById('videoList');
    if (!list) return;

    // Build a combined array from input files + directory files + dropped files
    const combined = [];
    if (videoInput && videoInput.files) {
        for (let i = 0; i < videoInput.files.length; i++) combined.push(videoInput.files[i]);
    }
    if (directoryInput && directoryInput.files) {
        for (let i = 0; i < directoryInput.files.length; i++) combined.push(directoryInput.files[i]);
    }
    if (window.droppedVideos && window.droppedVideos.length) {
        combined.push(...window.droppedVideos);
    }

    // Filter out removed items
    const visible = combined.filter(f => !removedVideoKeys.has(fileKey(f)));

    // Clear and render badges; keep section header persistent and update count
    list.innerHTML = '';
    const countEl = document.getElementById('selectedVideosCount');
    if (countEl) countEl.textContent = visible.length.toString();
    if (visible.length === 0) return;

    const badges = document.createElement('div');
    badges.className = 'video-badges';
    visible.forEach((file) => {
        const key = fileKey(file);
        const name = file.webkitRelativePath || file.name;
        const sizeMB = (file.size / (1024*1024)).toFixed(2);
        const badge = document.createElement('span');
        badge.className = 'video-badge';
        badge.title = name;
        badge.innerHTML = `
            <span class="name">${name}</span>
            <span class="size">- ${sizeMB} MB</span>
            <button type="button" class="remove" aria-label="Remove" title="Remove">×</button>
        `;
        badge.querySelector('.remove').addEventListener('click', function(){
            removedVideoKeys.add(key);
            listSelectedVideos();
        });
        badges.appendChild(badge);
    });
    list.appendChild(badges);
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
            
            // Disable all settings (after capturing form values below)
            
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
            // IMPORTANT: Disabled inputs are excluded from FormData. Build it before disabling.
            const formData = new FormData(form);
            // Ensure topPreviewCount is included even if control becomes disabled
            const topPreviewSlider = document.getElementById('topPreviewCount');
            if (topPreviewSlider) {
                formData.set('topPreviewCount', topPreviewSlider.value);
            }
            // Now it is safe to disable settings
            disableSettings(true);
            
            // Remove confidenceThreshold from formData since we removed the slider
            formData.delete('confidenceThreshold');
            
            // Handle reference images properly (respect removed badges)
            const referenceInput = document.getElementById('reference_images');
            if (referenceInput && referenceInput.files.length > 0) {
                // Clear the existing reference_images entries
                formData.delete('reference_images');
                // For multiple file uploads, we need to append all files individually
                for (let i = 0; i < referenceInput.files.length; i++) {
                    const f = referenceInput.files[i];
                    if (!removedReferenceKeys.has(fileKey(f))) {
                        formData.append('reference_images', f);
                    }
                }
            }
            
            // No negative references anymore
            
            // Handle video files properly (respect removed badges)
            const videoInput = document.getElementById('videos');
            if (videoInput && videoInput.files.length > 0) {
                // Clear the existing videos entries
                formData.delete('videos');
                // For multiple file uploads, we need to append all files individually
                for (let i = 0; i < videoInput.files.length; i++) {
                    const f = videoInput.files[i];
                    if (!removedVideoKeys.has(fileKey(f))) {
                        formData.append('videos', f);
                    }
                }
            }
            
            // Handle directory uploads properly
            const directoryInput = document.getElementById('videoDirectory');
            if (directoryInput && directoryInput.files.length > 0) {
                // Clear the existing videoDirectory entries
                formData.delete('videoDirectory');
                // For directory uploads, we need to append all files individually
                for (let i = 0; i < directoryInput.files.length; i++) {
                    const f = directoryInput.files[i];
                    if (!removedVideoKeys.has(fileKey(f))) {
                        formData.append('videoDirectory', f);
                    }
                }
            }
            
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

        // Comprehensive reset: clear files, settings, and UI state
        form.addEventListener('reset', function(e){
            e.preventDefault();
            // Clear file inputs
            const refIn = document.getElementById('reference_images');
            const vidIn = document.getElementById('videos');
            const dirIn = document.getElementById('videoDirectory');
            if (refIn) refIn.value = '';
            if (vidIn) vidIn.value = '';
            if (dirIn) dirIn.value = '';
            // Clear previews and lists
            const refPrev = document.getElementById('referencePreview');
            if (refPrev) refPrev.innerHTML = '';
            const vidList = document.getElementById('videoList');
            if (vidList) vidList.innerHTML = '';
            const selCount = document.getElementById('selectedVideosCount');
            if (selCount) selCount.textContent = '0';
            // Clear dropped and removed state
            try { droppedReferences = []; } catch(_) {}
            try { window.droppedVideos = []; } catch(_) {}
            removedVideoKeys = new Set();
            // Reset sliders and selects to defaults
            const frameInterval = document.getElementById('frameInterval');
            if (frameInterval) frameInterval.value = '1.0';
            const scanningMode = document.getElementById('scanningMode');
            if (scanningMode) scanningMode.value = 'balanced';
            const topPreviewSlider = document.getElementById('topPreviewCount');
            const topPreviewValue = document.getElementById('topPreviewCountValue');
            if (topPreviewSlider) topPreviewSlider.value = '5';
            if (topPreviewValue) topPreviewValue.textContent = '5';
            // Clear localStorage settings
            try { localStorage.removeItem('frameFinderSettings'); } catch(_) {}
            try { localStorage.removeItem('frameFinderTopPreviewCount'); } catch(_) {}
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

    // Removed negative references UI

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

    // --- Drag & drop support ---
    let droppedReferences = [];
    // Expose droppedVideos globally so listSelectedVideos can read it
    window.droppedVideos = [];

    function setupDropzone(zoneId, onFiles) {
        const dz = document.getElementById(zoneId);
        if (!dz) return;
        const input = dz.querySelector('input[type="file"]');
        function stop(e){ e.preventDefault(); e.stopPropagation(); }
        ['dragenter','dragover','dragleave','drop'].forEach(evt => dz.addEventListener(evt, stop));
        ['dragenter','dragover'].forEach(evt => dz.addEventListener(evt, () => dz.classList.add('dragover')));
        ;['dragleave','drop'].forEach(evt => dz.addEventListener(evt, () => dz.classList.remove('dragover')));
        dz.addEventListener('drop', (e) => {
            const files = Array.from(e.dataTransfer.files || []);
            onFiles(files);
        });
        // Do NOT bind input change here; existing listeners handle previews/listing.
    }

    function addRefFiles(files){
        const imgs = files.filter(f => f.type.startsWith('image/'));
        if (imgs.length === 0) return;
        droppedReferences = droppedReferences.concat(imgs);
        previewReferenceImages();
    }
    function addVideoFiles(files){
        const vids = files.filter(f => f.type === 'video/mp4' || f.name.toLowerCase().endsWith('.mp4'));
        if (vids.length === 0) return;
        window.droppedVideos = (window.droppedVideos || []).concat(vids);
        listSelectedVideos();
    }

    setupDropzone('refDrop', addRefFiles);
    setupDropzone('vidDrop', addVideoFiles);

    // Override previews to include dropped files
    // No need to override reference preview now; previewReferenceImages handles dropped refs

    const _origListVideos = listSelectedVideos;
    window.listSelectedVideos = function(){ _origListVideos(); }

    // Hook FormData assembly to include dropped files too
    const form = document.getElementById('uploadForm');
    if (form) {
        const originalSubmitHandler = form.onsubmit; // not used; we add listener in handleFormSubmission
        form.addEventListener('submit', function(){
            // Inject dropped files by patching fetch FormData creation
            const origFetch = window.fetch;
            window.fetch = function(input, init){
                try {
                    if (init && init.body instanceof FormData) {
                        // Append dropped references (respect removed badges)
                        droppedReferences.forEach(f => {
                            if (!removedReferenceKeys.has(fileKey(f))) {
                                init.body.append('reference_images', f);
                            }
                        });
                        // Append dropped videos (respect removed badges)
                        (window.droppedVideos || []).forEach(f => {
                            if (!removedVideoKeys.has(fileKey(f))) {
                                init.body.append('videos', f);
                            }
                        });
                    }
                } catch(e){}
                return origFetch(input, init);
            };
        }, { once: true });
    }
});
