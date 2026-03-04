/**
 * Open-Cowork Chrome Extension — Content Script
 * Sprint 46: Chrome Extension Bridge
 *
 * Injected into every page to provide DOM access capabilities.
 * Listens for messages from the background service worker.
 */

// Listen for messages from background.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'getAccessibilityTree') {
    // buildAccessibilityTree is injected via executeScript from background.js
    sendResponse({ status: 'ready' });
  } else if (message.type === 'getPageText') {
    sendResponse({ text: document.body ? document.body.innerText : '' });
  } else if (message.type === 'ping') {
    sendResponse({ status: 'alive', url: window.location.href });
  }
  return true;
});

console.log('[Cowork Bridge] Content script loaded');
