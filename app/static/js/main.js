document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const predictBtn = document.getElementById('predictBtn');
    const fileInput = document.getElementById('fileInput');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const prediction = document.getElementById('prediction');
    const error = document.getElementById('error');

    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        handleFileUpload('/upload');
    });

    predictBtn.addEventListener('click', function() {
        if (!fileInput.files[0]) {
            showError('Please upload a file first');
            return;
        }
        handleFileUpload('/predict');
    });

    function handleFileUpload(endpoint) {
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        showLoading();
        hideError();
        hideResults();
        hidePrediction();

        fetch(endpoint, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.error) {
                showError(data.error);
            } else if (endpoint === '/predict') {
                showPrediction(data);
            } else {
                showResults(data);
            }
        })
        .catch(err => {
            hideLoading();
            showError('An error occurred while processing your request');
            console.error(err);
        });
    }

    function showPrediction(data) {
        const predictionResult = document.getElementById('predictionResult');
        predictionResult.innerHTML = `
            <div class="alert alert-info">
                <h4 class="alert-heading">Predicted Expenses</h4>
                <p class="mb-0">Next week's expenses: $${data.prediction.next_week_expenses.toFixed(2)}</p>
            </div>
        `;
        prediction.classList.remove('d-none');
    }

    function showResults(data) {
        const resultsDiv = document.getElementById('results');
        resultsDiv.classList.remove('d-none');
        
        // Display summary statistics
        const summaryStats = document.getElementById('summaryStats');
        summaryStats.innerHTML = `
            <div class="stat-item">
                <span class="stat-label">Total Transactions:</span>
                <span class="stat-value">${data.total_transactions}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Amount:</span>
                <span class="stat-value">$${data.total_amount.toFixed(2)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Average Transaction:</span>
                <span class="stat-value">$${data.avg_transaction.toFixed(2)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Transaction Std Dev:</span>
                <span class="stat-value">$${data.std_transaction.toFixed(2)}</span>
            </div>
        `;

        // Display time patterns
        const timePatterns = document.getElementById('timePatterns');
        timePatterns.innerHTML = `
            <h4>Time Patterns</h4>
            <div class="stat-item">
                <span class="stat-label">Weekend Average:</span>
                <span class="stat-value">$${data.time_patterns.weekend_vs_weekday.weekend_avg.toFixed(2)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Weekday Average:</span>
                <span class="stat-value">$${data.time_patterns.weekend_vs_weekday.weekday_avg.toFixed(2)}</span>
            </div>
        `;

        // Display client summary
        const clientStats = document.getElementById('clientStats');
        const clientItems = data.client_summary
            .map(client => `
                <div class="client-summary">
                    <div class="stat-item">
                        <span class="stat-label">Average Amount:</span>
                        <span class="stat-value">$${client.avg_amount.toFixed(2)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Total Spent:</span>
                        <span class="stat-value">$${client.total_spent.toFixed(2)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Transaction Count:</span>
                        <span class="stat-value">${client.transaction_count}</span>
                    </div>
                </div>
            `).join('');
        clientStats.innerHTML = clientItems;

        // Display monthly summary
        const monthlyStats = document.getElementById('monthlyStats');
        const monthlyItems = Object.entries(data.monthly_summary)
            .map(([month, stats]) => `
                <div class="stat-item">
                    <h6 class="mb-2">${month}</h6>
                    <div class="ms-3">
                        <div class="stat-item">
                            <span class="stat-label">Total Amount:</span>
                            <span class="stat-value">$${stats.total_amount.toFixed(2)}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Transactions:</span>
                            <span class="stat-value">${stats.transaction_count}</span>
                        </div>
                    </div>
                </div>
            `).join('');
        monthlyStats.innerHTML = monthlyItems;
    }

    function showLoading() {
        loading.classList.remove('d-none');
    }

    function hideLoading() {
        loading.classList.add('d-none');
    }

    function showError(message) {
        error.textContent = message;
        error.classList.remove('d-none');
    }

    function hideError() {
        error.classList.add('d-none');
    }

    function hideResults() {
        results.classList.add('d-none');
    }

    function hidePrediction() {
        prediction.classList.add('d-none');
    }
}); 