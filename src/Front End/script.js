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

// Call on every page load
window.addEventListener('DOMContentLoaded', () => {
    // Detect navigation type: “reload” vs “navigate” (link click)
    let navType = 'navigate';
    if (performance.getEntriesByType) {
        const [nav] = performance.getEntriesByType('navigation') || [];
        navType = nav ? nav.type : navType;
    } else if (performance.navigation) {
        navType = (performance.navigation.type === 1) ? 'reload' : 'navigate';
    }

    if (navType === 'reload') {
        // 1) True refresh: clear sessionStorage and wipe all inputs & outputs
        sessionStorage.clear();
        document.querySelectorAll('input, textarea').forEach(el => el.value = '');
        // example of wiping an output area:
        const status = document.getElementById('status-message');
        if (status) status.textContent = 'Waiting for credentials…';
        const propList = document.getElementById('prop-list');
        if (propList) propList.innerHTML = '';
    } else {
        // 2) Navigation between pages: restore saved values
        document.querySelectorAll('input, textarea').forEach(el => {
            const saved = sessionStorage.getItem(el.id);
            if (saved !== null) el.value = saved;
        });
    }

    // 3) On any change, always store the latest value in sessionStorage
    document.querySelectorAll('input, textarea').forEach(el => {
        el.addEventListener('input', () => {
            sessionStorage.setItem(el.id, el.value);
        });
    });
});

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


// ---------- File Menu Actions ----------

// Load a JSON or text file of English sentences as your KB
function loadKB() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.txt,.json';
    input.onchange = e => {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = () => {
        const lines = reader.result.split(/\r?\n/).filter(Boolean);
        // assume you have an array `kbSentences` and a display area
        kbSentences = lines;
        displayStatus(`Loaded KB: ${file.name} (${lines.length} statements)`);
        refreshKBDisplay();
      };
      reader.readAsText(file);
    };
    input.click();
  }
  
  // Save current KB to a JSON file
  function saveKB() {
    const blob = new Blob([JSON.stringify(kbSentences, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'knowledge_base.json';
    a.click();
    URL.revokeObjectURL(url);
    displayStatus('Knowledge base saved.');
  }
  
  // Clear the in-memory KB
  function clearKB() {
    if (!confirm('Clear all loaded statements?')) return;
    kbSentences = [];
    refreshKBDisplay();
    displayStatus('Knowledge base cleared.');
  }
  
  // ---------- Edit Menu Actions ----------
  
  // Clear the English-sentence input textarea
  function clearInput() {
    document.getElementById('input-area').value = '';
    displayStatus('Input cleared.');
  }
  
  // Clear the query/output display area
  function clearOutput() {
    document.getElementById('output-area').textContent = '';
    displayStatus('Output cleared.');
  }
  
  // Copy the output text to clipboard
  function copyOutput() {
    const out = document.getElementById('output-area').textContent;
    navigator.clipboard.writeText(out).then(
      () => displayStatus('Output copied to clipboard.'),
      () => displayStatus('Failed to copy output.')
    );
  }
  
  // ---------- Run Menu Actions ----------
  
  // Parse & assert all loaded statements, then report success
  function executeAll() {
    parser.assert_knowledge(kbSentences);
    displayStatus(`Asserted ${kbSentences.length} statements.`);
  }
  
  // Run whatever is in the query input field
  function runQuery() {
    const q = document.getElementById('query-input').value;
    if (!q.trim()) { displayStatus('No query entered.'); return; }
    try {
      const goal = parser.parse_query(q);
      const sols = kb.query_all(goal);
      document.getElementById('output-area').textContent = JSON.stringify(sols, null, 2);
      displayStatus(`Query returned ${sols.length} solution(s).`);
    } catch (err) {
      displayStatus('Query error: ' + err.message);
    }
  }
  
  // Quick syntax check without asserting to Prolog
  function validateSyntax() {
    const q = document.getElementById('query-input').value;
    try {
      parser.parse_query(q);
      displayStatus('Syntax OK.');
    } catch (err) {
      displayStatus('Syntax error: ' + err.message);
    }
  }
  
  // ---------- Help Menu Actions ----------
  
  function showAbout() {
    alert('Logic Analyzer v1.0\nBuilt with SemanticParser + PySwip');
  }
  
  function showDocumentation() {
    window.open('https://github.com/AndyFerns/Automated-Reasoning-Project#readme', '_blank');
  }
  
  // ---------- Utility: status bar updater ----------
  function displayStatus(msg) {
    document.getElementById('status-message').textContent = msg;
  }