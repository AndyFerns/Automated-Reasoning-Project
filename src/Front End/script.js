// State Persistence Helpers
function saveState() {
    try {
        localStorage.setItem('input1', document.getElementById('input1').value);
        localStorage.setItem('input2', document.getElementById('input2').value);
        localStorage.setItem('output', document.getElementById('output').value);
    } catch (e) {
        console.warn('Could not save state:', e);
    }
}
// Loads latest saved state
function loadState() {
    try {
        const input1 = document.getElementById('input1');
        const input2 = document.getElementById('input2');
        const output = document.getElementById('output');
        const v1 = localStorage.getItem('input1');
        const v2 = localStorage.getItem('input2');
        const v3 = localStorage.getItem('output');
        if (v1 !== null) input1.value = v1;
        if (v2 !== null) input2.value = v2; 
        if (v3 !== null) output.value = v3;
    } catch (e) {
        console.warn('Could not load state:', e);
    }
}

// Save state before navigating away
window.addEventListener('beforeunload', saveState);


// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// Make panels resizable
function setupResizable(resizeHandleId, leftPanelSelector, rightPanelSelector) {
    const resizeHandle = document.getElementById(resizeHandleId);
    const leftPanel = document.querySelector(leftPanelSelector);
    const rightPanel = document.querySelector(rightPanelSelector);
    const container = document.querySelector('.container');

    let isResizing = false;
    let startX, leftWidth, rightWidth;

    resizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX;
        leftWidth = leftPanel.offsetWidth;
        rightWidth = rightPanel.offsetWidth;
        document.body.style.cursor = 'col-resize';
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        e.preventDefault();
    });

    function handleMouseMove(e) {
        if (!isResizing) return;
        
        const dx = e.clientX - startX;
        const containerWidth = container.offsetWidth;
        
        // Calculate new widths in percentages
        const newLeftWidth = ((leftWidth + dx) / containerWidth) * 100;
        const newRightWidth = ((rightWidth - dx) / containerWidth) * 100;
        
        // Apply minimum width constraints
        if (newLeftWidth > 10 && newRightWidth > 10) {
            leftPanel.style.flex = `0 0 ${newLeftWidth}%`;
            rightPanel.style.flex = `0 0 ${newRightWidth}%`;
        }
    }

    function handleMouseUp() {
        isResizing = false;
        document.body.style.cursor = '';
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
    }
}

// Status message helper
function setStatus(message) {
    document.getElementById('status-message').textContent = message;
}

// API Functions
async function assertStatement(sentence) {
    try {
        setStatus('Asserting statement...');
        const response = await fetch(`${API_BASE_URL}/assert`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ sentence }),
            credentials: 'include'
        });
        
        const data = await response.json();
        
        if (response.ok) {
            setStatus('Statement asserted successfully');
            return data.predicate;
        } else {
            throw new Error(data.error || 'Failed to assert statement');
        }
    } catch (error) {
        setStatus(`Error: ${error.message}`);
        console.error('Assert error:', error);
        throw error;
    }
}

async function queryKnowledgeBase(rawQuery) {
    try {
        setStatus('Querying knowledge base...');
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ query: rawQuery })
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to execute query');
        }
        setStatus('Query executed successfully');
        return data; // { parsed_query, results }
    } catch (error) {
        setStatus(`Error: ${error.message}`);
        console.error('Query error:', error);
        throw error;
    }
}


async function saveKnowledgeBase(filename = 'knowledge_base.pl') {
    try {
        setStatus('Saving knowledge base...');
        const response = await fetch(`${API_BASE_URL}/save`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ filename }),
            credentials: 'include'
        });
        
        const data = await response.json();
        
        if (response.ok) {
            setStatus(`Saved to ${filename}`);
            return data.message;
        } else {
            throw new Error(data.error || 'Failed to save knowledge base');
        }
    } catch (error) {
        setStatus(`Error: ${error.message}`);
        console.error('Save error:', error);
        throw error;
    }
}

async function exportToNeo4j(uri, user, password) {
    try {
        setStatus('Exporting to Neo4j...');
        const response = await fetch(`${API_BASE_URL}/export_neo4j`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ uri, user, password }),
            credentials: 'include'
        });
        
        const data = await response.json();
        
        if (response.ok) {
            setStatus('Exported to Neo4j successfully');
            return data.message;
        } else {
            throw new Error(data.error || 'Failed to export to Neo4j');
        }
    } catch (error) {
        setStatus(`Error: ${error.message}`);
        console.error('Export error:', error);
        throw error;
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Save previous state
    loadState();

    // Set up both resize handles
    setupResizable('resize1', '.panel:nth-child(1)', '.panel:nth-child(3)');
    setupResizable('resize2', '.panel:nth-child(3)', '.panel:nth-child(5)');

    // UI Event Handlers
    document.querySelectorAll('.assert-btn').forEach(button => {
        button.addEventListener('click', async function() {
            const panel = this.closest('.panel');
            const textarea = panel.querySelector('textarea');
            const text = textarea.value.trim();
            
            if (text) {
                try {
                    const result = await assertStatement(text);
                    output.value += `Asserted: ${JSON.stringify(result)}\n\n`;
                    // textarea.value = ''; // Clear input after asserting
                } catch (error) {
                    output.value += `Error: ${error.message}\n\n`;
                }
            }
        });
    });

    // Full snippet in context:
    document.querySelectorAll('.query-btn').forEach(button => {
        button.addEventListener('click', async () => {
            const panel = button.closest('.panel');
            const input = panel.querySelector('textarea');
        
            // locate output box
            let output = panel.querySelector('textarea.output');
            if (!output) output = document.getElementById('output');
        
            const rawQuery = input.value.trim();
            if (!rawQuery) return;
        
            output.value += `> Raw: ${rawQuery}\n`;
        
            try {
                const { parsed_query, results } = await queryKnowledgeBase(rawQuery);
                output.value += `> Parsed: ${parsed_query}\n`;
        
                if (Array.isArray(results) && results.length) {
                    results.forEach(r => output.value += JSON.stringify(r) + '\n');
                    output.value += 'True \n';      
                } else {
                    output.value += 'No results found\n';
                }
            } catch (err) {
                    output.value += `Error: ${err.message}\n`;
            }
            output.value += '\n';
            output.scrollTop = output.scrollHeight;
        });
    });
      
    document.querySelector('.save-btn').addEventListener('click', async function() {
        const filename = prompt('Enter filename (default: knowledge_base.pl):', 'knowledge_base.pl');
        if (filename !== null) { // User didn't press cancel
            try {
                await saveKnowledgeBase(filename);
                output.value += `Knowledge base saved to ${filename}\n\n`;
            } catch (error) {
                output.value += `Error: ${error.message}\n\n`;
            }
        }
    });

    // Neo4j Export Modal
    const exportModal = document.getElementById('export-modal');
    const exportBtn = document.querySelector('.export-btn');
    const closeModal = document.querySelector('.close-modal');
    const cancelExport = document.getElementById('cancel-export');
    const confirmExport = document.getElementById('confirm-export');
    const output = document.getElementById('output');

    exportBtn.addEventListener('click', () => {
        exportModal.style.display = 'flex';
    });

    closeModal.addEventListener('click', () => {
        exportModal.style.display = 'none';
    });

    cancelExport.addEventListener('click', () => {
        exportModal.style.display = 'none';
    });

    confirmExport.addEventListener('click', async () => {
        const uri = document.getElementById('neo4j-uri').value;
        const user = document.getElementById('neo4j-user').value;
        const password = document.getElementById('neo4j-password').value;
        
        if (uri && user && password) {
            try {
                await exportToNeo4j(uri, user, password);
                output.value += `Exported to Neo4j at ${uri}\n\n`;
                exportModal.style.display = 'none';
            } catch (error) {
                output.value += `Error: ${error.message}\n\n`;
            }
        } else {
            alert('Please fill in all fields');
        }
    });

    // Close modal when clicking outside
    exportModal.addEventListener('click', (e) => {
        if (e.target === exportModal) {
            exportModal.style.display = 'none';
        }
    });

    // Initialize with some text
    // only show the welcome text if we had nothing saved
    if (!localStorage.getItem('output')) {
        output.value = "Knowledge Base Interface Ready\n\n";
    }
});