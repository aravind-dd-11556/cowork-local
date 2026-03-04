/**
 * Open-Cowork Chrome Extension — Content Script
 * Sprint 46: Chrome Extension Bridge
 *
 * Injected into every page to provide DOM access capabilities.
 * Listens for messages from the background service worker.
 */

// Listen for messages from background.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Validate message structure
  if (!message || typeof message !== 'object') {
    sendResponse({ success: false, error: 'Invalid message format' });
    return true;
  }

  try {
    if (message.type === 'getAccessibilityTree') {
      // buildAccessibilityTree is injected via executeScript from background.js
      sendResponse({ status: 'ready', success: true });
    } else if (message.type === 'getPageText') {
      const text = document.body ? document.body.innerText : '';
      sendResponse({ text, success: true });
    } else if (message.type === 'ping') {
      sendResponse({ status: 'alive', url: window.location.href, success: true });
    } else {
      sendResponse({ success: false, error: 'Unknown message type' });
    }
  } catch (e) {
    console.error('[Cowork Bridge] Error handling message:', e);
    sendResponse({ success: false, error: e.message });
  }

  return true;
});

console.log('[Cowork Bridge] Content script loaded');
