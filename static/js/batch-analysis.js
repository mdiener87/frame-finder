// batch-analysis.js - Handles batch analysis functionality

class BatchAnalyzer {
    constructor() {
        this.analysisId = null;
        this.pollingInterval = null;
    }
    
    async startAnalysis(referenceFile, videoFiles) {
        // Create FormData for file uploads
        const formData = new FormData();
        
        // Add reference file
        formData.append('reference', referenceFile);
        
        // Add video files
        videoFiles.forEach((file, index) => {
            formData.append(`video_${index}`, file);
        });
        
        try {
            const response = await fetch('/api/batch-analyze', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to start analysis');
            }
            
            this.analysisId = data.analysis_id;
            return data;
        } catch (error) {
            console.error('Analysis start error:', error);
            throw error;
        }
    }
    
    async getResults(analysisId) {
        try {
            const response = await fetch(`/api/analysis/${analysisId}`);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to get results');
            }
            
            return data;
        } catch (error) {
            console.error('Results fetch error:', error);
            throw error;
        }
    }
    
    startPolling(analysisId, callback) {
        this.analysisId = analysisId;
        this.stopPolling(); // Stop any existing polling
        
        this.pollingInterval = setInterval(async () => {
            try {
                const results = await this.getResults(analysisId);
                callback(null, results);
                
                // Stop polling if analysis is complete or failed
                if (results.status === 'completed' || results.status === 'failed') {
                    this.stopPolling();
                }
            } catch (error) {
                callback(error, null);
                this.stopPolling();
            }
        }, 2000); // Poll every 2 seconds
    }
    
    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }
    
    formatResults(results) {
        if (!results || results.status !== 'completed') {
            return '<p>Analysis not completed yet.</p>';
        }
        
        let html = `
        <div class="row">
            <div class="col-md-12">
                <h6>Analysis Summary</h6>
                <p>Processed ${Object.keys(results.results).length} videos.</p>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <h6>Results by Video</h6>
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Video</th>
                                <th>Max Similarity</th>
                                <th>Matches Found</th>
                                <th>Confidence</th>
                                <th>Likely Match</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        // Sort videos by likelihood
        const sortedVideos = Object.entries(results.comparison).sort((a, b) => {
            // Sort by is_likely_match first (true first), then by max_similarity
            if (a[1].is_likely_match && !b[1].is_likely_match) return -1;
            if (!a[1].is_likely_match && b[1].is_likely_match) return 1;
            return b[1].max_similarity - a[1].max_similarity;
        });
        
        for (const [videoName, comparison] of sortedVideos) {
            const result = Object.values(results.results).find(r => 
                r.video_name === videoName || 
                (r.video_path && r.video_path.endsWith(videoName))
            ) || {};
            
            const maxSimilarity = comparison.max_similarity || result.max_similarity || 0;
            const matchesCount = comparison.matches_count || result.matches_count || 0;
            
            const rowClass = comparison.is_likely_match ? 'table-success' : '';
            const badgeClass = comparison.is_likely_match ? 'bg-success' : 'bg-secondary';
            const badgeText = comparison.is_likely_match ? 'YES' : 'NO';
            
            html += `
            <tr class="${rowClass}">
                <td>${videoName}</td>
                <td>${(maxSimilarity * 100).toFixed(2)}%</td>
                <td>${matchesCount}</td>
                <td>${comparison.confidence_level.charAt(0).toUpperCase() + comparison.confidence_level.slice(1)}</td>
                <td><span class="badge ${badgeClass}">${badgeText}</span></td>
            </tr>
            `;
        }
        
        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <h6>Detailed Analysis</h6>
                <div class="alert alert-info">
                    <h6>How we determined this:</h6>
                    <ul>
        `;
        
        // Add reasoning for each video
        for (const [videoName, comparison] of Object.entries(results.comparison)) {
            html += `<li><strong>${videoName}:</strong> ${comparison.reasoning}</li>`;
        }
        
        html += `
                    </ul>
                    <p class="mb-0">Videos marked as "YES" are most likely to contain the reference image.</p>
                </div>
            </div>
        </div>
        `;
        
        return html;
    }
}

// Make it available globally
window.BatchAnalyzer = BatchAnalyzer;