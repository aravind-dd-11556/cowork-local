/**
 * Open-Cowork Chrome Extension — Popup Logic
 * Sprint 46: Chrome Extension Bridge
 */

let isConnected = false;

function updateUI(connected, agentUrl) {
  isConnected = connected;
  const status = document.getElementById('status');
  const dot = document.getElementById('dot');
  const statusText = document.getElementById('statusText');
  const btn = document.getElementById('actionBtn');
  const info = document.getElementById('info');

  if (connected) {
    status.className = 'status connected';
    dot.className = 'dot green';
    statusText.textContent = 'Connected';
    btn.textContent = 'Disconnect';
    btn.className = 'btn-disconnect';
  } else {
    status.className = 'status disconnected';
    dot.className = 'dot red';
    statusText.textContent = 'Disconnected';
    btn.textContent = 'Connect to Agent';
    btn.className = 'btn-connect';
  }

  if (agentUrl) {
    info.textContent = `Agent URL: ${agentUrl}`;
  }
}

function toggleConnection() {
  const messageType = isConnected ? 'disconnect' : 'connect';
  chrome.runtime.sendMessage({ type: messageType }, (response) => {
    // Status will be updated via status_update message
    setTimeout(refreshStatus, 500);
  });
}

function refreshStatus() {
  chrome.runtime.sendMessage({ type: 'get_status' }, (response) => {
    if (response) {
      updateUI(response.connected, response.agentUrl);
    }
  });
}

// Listen for status updates from background
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'status_update') {
    updateUI(message.connected, message.agentUrl);
  }
});

// Get initial status
refreshStatus();
