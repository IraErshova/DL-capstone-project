document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('Please select a file to upload');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading state
    document.getElementById('loading').classList.remove('d-none');
    document.getElementById('results').classList.add('d-none');
    document.getElementById('error').classList.add('d-none');
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'An error occurred while processing the file');
        }
        
        displayResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        document.getElementById('loading').classList.add('d-none');
    }
});

function displayResults(data) {
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
            <span class="stat-label">Total Income:</span>
            <span class="stat-value">$${data.total_income.toFixed(2)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Total Expenses:</span>
            <span class="stat-value">$${data.total_expenses.toFixed(2)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Net Cash Flow:</span>
            <span class="stat-value">$${data.net_flow.toFixed(2)}</span>
        </div>
    `;
    
    // Display category statistics
    const categoryStats = document.getElementById('categoryStats');
    const categoryItems = Object.entries(data.categories)
        .map(([category, count]) => `
            <div class="stat-item">
                <span class="stat-label">${category}:</span>
                <span class="stat-value">${count}</span>
            </div>
        `).join('');
    categoryStats.innerHTML = categoryItems;
    
    // Display monthly trends
    const monthlyTrends = document.getElementById('monthlyTrends');
    const months = Object.keys(data.monthly_summary);
    const amounts = Object.values(data.monthly_summary);
    
    const trace = {
        x: months,
        y: amounts,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Monthly Amount',
        line: {
            color: '#0d6efd',
            width: 2
        },
        marker: {
            size: 8,
            color: '#0d6efd'
        }
    };
    
    const layout = {
        title: 'Monthly Transaction Trends',
        xaxis: {
            title: 'Month',
            tickangle: -45
        },
        yaxis: {
            title: 'Amount ($)'
        },
        margin: {
            l: 50,
            r: 50,
            b: 100,
            t: 50,
            pad: 4
        }
    };
    
    Plotly.newPlot(monthlyTrends, [trace], layout);
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.classList.remove('d-none');
} 