/**
 * TB Simulation Web App - Frontend JavaScript
 * Handles script execution, real-time output, and WebSocket communication
 */

// Global variables
let socket;
let isExecuting = false;
let currentExecution = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeSocket();
    setupExecutionHandlers();
});

function initializeSocket() {
    // Connect to Socket.IO server
    socket = io();
    
    socket.on('connect', function() {
        console.log('Connected to server');
        updateConnectionStatus('Connected', 'success');
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
        updateConnectionStatus('Disconnected', 'danger');
    });
    
    socket.on('output', function(data) {
        console.log('Received output:', data);
        appendOutput(data.line);
    });
    
    socket.on('execution_complete', function(data) {
        handleExecutionComplete(data);
    });
    
    socket.on('execution_error', function(data) {
        handleExecutionError(data);
    });
}

function updateConnectionStatus(message, type) {
    const statusElement = document.getElementById('connection-status');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = `badge bg-${type}`;
    }
}

function setupExecutionHandlers() {
    // Setup execution button handlers
    const executeBtn = document.getElementById('execute-btn');
    const stopBtn = document.getElementById('stop-btn');
    
    if (executeBtn) {
        executeBtn.addEventListener('click', executeScript);
    }
    
    if (stopBtn) {
        stopBtn.addEventListener('click', stopExecution);
    }
}

function executeScript() {
    console.log('Execute script called, currentScript:', currentScript);
    
    if (!currentScript) {
        showAlert('No script selected for execution', 'warning');
        return;
    }
    
    if (isExecuting) {
        showAlert('Script is already executing', 'warning');
        return;
    }
    
    // Clear previous output
    clearOutput();
    
    // Show execution controls
    document.getElementById('execute-btn').style.display = 'none';
    document.getElementById('stop-btn').style.display = 'inline-block';
    
    // Start execution
    isExecuting = true;
    appendOutput('Starting script execution...', 'info');
    
    console.log('Sending execute request for:', currentScript.path);
    
    fetch('/api/execute', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            script_path: currentScript.path,
            parameters: {} // Could be extended to support parameters
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Execute response:', data);
        if (data.status === 'started') {
            appendOutput('Script execution started successfully', 'success');
        } else {
            throw new Error(data.message || 'Failed to start execution');
        }
    })
    .catch(error => {
        console.error('Execution error:', error);
        appendOutput(`Error: ${error.message}`, 'error');
        handleExecutionComplete({ exit_code: 1 });
    });
}

function stopExecution() {
    if (!isExecuting) {
        return;
    }
    
    fetch('/api/stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'stopped') {
            appendOutput('Script execution stopped by user', 'warning');
            handleExecutionComplete({ exit_code: -1 });
        }
    })
    .catch(error => {
        console.error('Stop error:', error);
        appendOutput(`Error stopping execution: ${error.message}`, 'error');
    });
}

function handleExecutionComplete(data) {
    isExecuting = false;
    
    // Update UI
    document.getElementById('execute-btn').style.display = 'inline-block';
    document.getElementById('stop-btn').style.display = 'none';
    
    // Show completion message
    if (data.exit_code === 0) {
        appendOutput('Script execution completed successfully', 'success');
    } else if (data.exit_code === -1) {
        appendOutput('Script execution stopped by user', 'warning');
    } else {
        appendOutput(`Script execution completed with exit code: ${data.exit_code}`, 'error');
    }
    
    // Show execution summary
    if (data.output && data.output.length > 0) {
        appendOutput(`\n--- Execution Summary ---`, 'info');
        appendOutput(`Total output lines: ${data.output.length}`, 'info');
    }
}

function handleExecutionError(data) {
    isExecuting = false;
    
    // Update UI
    document.getElementById('execute-btn').style.display = 'inline-block';
    document.getElementById('stop-btn').style.display = 'none';
    
    // Show error
    appendOutput(`Execution error: ${data.error}`, 'error');
}

function appendOutput(text, type = 'normal') {
    const outputDiv = document.getElementById('execution-output');
    if (!outputDiv) {
        console.log('Output div not found');
        return;
    }
    
    console.log('Appending output:', text, 'Type:', type);
    
    // Create new line element
    const lineDiv = document.createElement('div');
    lineDiv.className = `output-line ${type}`;
    
    // Add timestamp
    const timestamp = new Date().toLocaleTimeString();
    lineDiv.innerHTML = `<span class="text-muted">[${timestamp}]</span> ${escapeHtml(text)}`;
    
    // Add to output
    outputDiv.appendChild(lineDiv);
    
    // Auto-scroll to bottom
    outputDiv.scrollTop = outputDiv.scrollHeight;
    
    // Limit output lines to prevent memory issues
    const lines = outputDiv.querySelectorAll('.output-line');
    if (lines.length > 1000) {
        lines[0].remove();
    }
}

function clearOutput() {
    const outputDiv = document.getElementById('execution-output');
    if (outputDiv) {
        outputDiv.innerHTML = '<div class="text-muted">Waiting for script execution...</div>';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showAlert(message, type) {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to page
    document.body.appendChild(alertDiv);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Utility functions for script execution
function showExecutionPanel() {
    const modal = new bootstrap.Modal(document.getElementById('executionModal'));
    modal.show();
}

function executeSelectedScript() {
    console.log('Execute selected script called, currentScript:', currentScript);
    
    if (currentScript) {
        // Close the details modal
        const detailsModal = bootstrap.Modal.getInstance(document.getElementById('scriptDetailsModal'));
        if (detailsModal) {
            detailsModal.hide();
        }
        
        // Show execution modal
        showExecutionPanel();
        
        // Set up execution details
        const scriptDetails = document.getElementById('script-details');
        if (scriptDetails) {
            scriptDetails.innerHTML = `
                <h6>${currentScript.name}</h6>
                <p class="small text-muted">${currentScript.description || 'No description'}</p>
                <p class="small"><strong>Path:</strong> ${currentScript.path}</p>
                <p class="small"><strong>Category:</strong> ${currentScript.category}</p>
            `;
        }
    } else {
        console.log('No currentScript available');
        showAlert('No script selected', 'warning');
    }
}

// Export functions for global access
window.executeScript = executeScript;
window.stopExecution = stopExecution;
window.showExecutionPanel = showExecutionPanel;
window.executeSelectedScript = executeSelectedScript;
