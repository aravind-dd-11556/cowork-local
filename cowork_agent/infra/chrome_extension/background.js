/**
 * Open-Cowork Chrome Extension — Background Service Worker
 * Sprint 46: Chrome Extension Bridge
 *
 * Acts as a WebSocket server that accepts JSON-RPC 2.0 calls from
 * the Open-Cowork agent and translates them into Chrome API calls.
 */

// ── State ────────────────────────────────────────────────────────
let ws = null;
let agentConnected = false;
let debuggerAttached = {};  // tabId -> boolean
let consoleBuffer = {};     // tabId -> messages[]
let networkBuffer = {};     // tabId -> requests[]
const AGENT_WS_URL = 'ws://localhost:9222';

// ── WebSocket Connection ─────────────────────────────────────────

function connectToAgent() {
  try {
    ws = new WebSocket(AGENT_WS_URL);

    ws.onopen = () => {
      agentConnected = true;
      console.log('[Cowork Bridge] Connected to agent');
      updatePopupStatus(true);
    };

    ws.onmessage = async (event) => {
      try {
        const request = JSON.parse(event.data);
        if (request.jsonrpc !== '2.0') return;

        const result = await handleRPCRequest(request);

        if (request.id !== undefined) {
          ws.send(JSON.stringify({
            jsonrpc: '2.0',
            id: request.id,
            result: result,
          }));
        }
      } catch (error) {
        if (event.data && JSON.parse(event.data).id !== undefined) {
          ws.send(JSON.stringify({
            jsonrpc: '2.0',
            id: JSON.parse(event.data).id,
            error: { code: -32603, message: error.message },
          }));
        }
      }
    };

    ws.onclose = () => {
      agentConnected = false;
      console.log('[Cowork Bridge] Disconnected from agent');
      updatePopupStatus(false);
    };

    ws.onerror = (error) => {
      console.error('[Cowork Bridge] WebSocket error:', error);
      agentConnected = false;
    };

  } catch (error) {
    console.error('[Cowork Bridge] Connection failed:', error);
  }
}

function disconnectFromAgent() {
  if (ws) {
    ws.close();
    ws = null;
  }
  agentConnected = false;
  updatePopupStatus(false);
}

// ── RPC Request Handler ──────────────────────────────────────────

async function handleRPCRequest(request) {
  const { method, params } = request;

  switch (method) {
    case 'browser/navigate':
      return await handleNavigate(params);
    case 'browser/screenshot':
      return await handleScreenshot(params);
    case 'browser/getAccessibilityTree':
      return await handleGetAccessibilityTree(params);
    case 'browser/findElements':
      return await handleFindElements(params);
    case 'browser/formInput':
      return await handleFormInput(params);
    case 'browser/performAction':
      return await handlePerformAction(params);
    case 'browser/executeScript':
      return await handleExecuteScript(params);
    case 'browser/getPageText':
      return await handleGetPageText(params);
    case 'browser/readConsole':
      return await handleReadConsole(params);
    case 'browser/readNetwork':
      return await handleReadNetwork(params);
    case 'browser/resize':
      return await handleResize(params);
    case 'browser/getTabs':
      return await handleGetTabs(params);
    case 'browser/createTab':
      return await handleCreateTab(params);
    default:
      throw new Error(`Unknown method: ${method}`);
  }
}

// ── Navigation ───────────────────────────────────────────────────

async function handleNavigate({ tabId, url }) {
  if (url === 'back') {
    await chrome.tabs.goBack(tabId);
    return { success: true, url: 'back' };
  }
  if (url === 'forward') {
    await chrome.tabs.goForward(tabId);
    return { success: true, url: 'forward' };
  }
  const tab = await chrome.tabs.update(tabId, { url });
  return { success: true, url: tab.url, title: tab.title };
}

// ── Screenshot ───────────────────────────────────────────────────

async function handleScreenshot({ tabId }) {
  const tab = await chrome.tabs.get(tabId);
  const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
    format: 'png',
  });
  return { success: true, data: dataUrl, format: 'png' };
}

// ── Accessibility Tree ───────────────────────────────────────────

async function handleGetAccessibilityTree({ tabId }) {
  const results = await chrome.scripting.executeScript({
    target: { tabId },
    func: buildAccessibilityTree,
  });
  const tree = results && results[0] ? results[0].result : null;
  return { success: true, tree };
}

// ── Find Elements ────────────────────────────────────────────────

async function handleFindElements({ tabId, query }) {
  const results = await chrome.scripting.executeScript({
    target: { tabId },
    func: findElementsInPage,
    args: [query],
  });
  const elements = results && results[0] ? results[0].result : [];
  return { success: true, elements };
}

// ── Form Input ───────────────────────────────────────────────────

async function handleFormInput({ tabId, ref, value }) {
  const results = await chrome.scripting.executeScript({
    target: { tabId },
    func: setFormValueInPage,
    args: [ref, value],
  });
  const result = results && results[0] ? results[0].result : { success: false };
  return result;
}

// ── Mouse/Keyboard Actions via CDP ──────────────────────────────

async function handlePerformAction({ tabId, action, ...params }) {
  await ensureDebuggerAttached(tabId);

  switch (action) {
    case 'left_click':
    case 'right_click':
    case 'double_click':
    case 'triple_click': {
      const [x, y] = params.coordinate || [0, 0];
      const button = action === 'right_click' ? 'right' : 'left';
      const clickCount = action === 'double_click' ? 2 : action === 'triple_click' ? 3 : 1;

      await chrome.debugger.sendCommand({ tabId }, 'Input.dispatchMouseEvent', {
        type: 'mousePressed', x, y, button, clickCount,
      });
      await chrome.debugger.sendCommand({ tabId }, 'Input.dispatchMouseEvent', {
        type: 'mouseReleased', x, y, button, clickCount,
      });
      return { success: true, action, coordinate: [x, y] };
    }

    case 'type': {
      const text = params.text || '';
      for (const char of text) {
        await chrome.debugger.sendCommand({ tabId }, 'Input.dispatchKeyEvent', {
          type: 'keyDown', text: char,
        });
        await chrome.debugger.sendCommand({ tabId }, 'Input.dispatchKeyEvent', {
          type: 'keyUp', text: char,
        });
      }
      return { success: true, action, text };
    }

    case 'key': {
      const keys = (params.text || '').split(' ');
      for (const key of keys) {
        await chrome.debugger.sendCommand({ tabId }, 'Input.dispatchKeyEvent', {
          type: 'keyDown', key,
        });
        await chrome.debugger.sendCommand({ tabId }, 'Input.dispatchKeyEvent', {
          type: 'keyUp', key,
        });
      }
      return { success: true, action, keys };
    }

    case 'scroll': {
      const [sx, sy] = params.coordinate || [640, 450];
      const deltaY = params.scroll_direction === 'up' ? -100 * (params.scroll_amount || 3)
                    : params.scroll_direction === 'down' ? 100 * (params.scroll_amount || 3) : 0;
      const deltaX = params.scroll_direction === 'left' ? -100 * (params.scroll_amount || 3)
                    : params.scroll_direction === 'right' ? 100 * (params.scroll_amount || 3) : 0;

      await chrome.debugger.sendCommand({ tabId }, 'Input.dispatchMouseEvent', {
        type: 'mouseWheel', x: sx, y: sy, deltaX, deltaY,
      });
      return { success: true, action, direction: params.scroll_direction };
    }

    case 'hover': {
      const [hx, hy] = params.coordinate || [0, 0];
      await chrome.debugger.sendCommand({ tabId }, 'Input.dispatchMouseEvent', {
        type: 'mouseMoved', x: hx, y: hy,
      });
      return { success: true, action, coordinate: [hx, hy] };
    }

    case 'screenshot': {
      return await handleScreenshot({ tabId });
    }

    case 'wait': {
      const duration = Math.min(params.duration || 1, 30) * 1000;
      await new Promise(resolve => setTimeout(resolve, duration));
      return { success: true, action, duration: params.duration };
    }

    default:
      return { success: true, action, note: 'Action simulated' };
  }
}

// ── JavaScript Execution ─────────────────────────────────────────

async function handleExecuteScript({ tabId, code }) {
  const results = await chrome.scripting.executeScript({
    target: { tabId },
    func: (jsCode) => {
      try {
        return { success: true, result: eval(jsCode) };
      } catch (e) {
        return { success: false, error: e.message };
      }
    },
    args: [code],
  });
  return results && results[0] ? results[0].result : { success: false, error: 'No result' };
}

// ── Page Text ────────────────────────────────────────────────────

async function handleGetPageText({ tabId }) {
  const results = await chrome.scripting.executeScript({
    target: { tabId },
    func: () => document.body ? document.body.innerText : '',
  });
  const text = results && results[0] ? results[0].result : '';
  return { success: true, text };
}

// ── Console Messages ─────────────────────────────────────────────

async function handleReadConsole({ tabId }) {
  const messages = consoleBuffer[tabId] || [];
  return { success: true, messages };
}

// ── Network Requests ─────────────────────────────────────────────

async function handleReadNetwork({ tabId }) {
  const requests = networkBuffer[tabId] || [];
  return { success: true, requests };
}

// ── Window Resize ────────────────────────────────────────────────

async function handleResize({ tabId, width, height }) {
  const tab = await chrome.tabs.get(tabId);
  await chrome.windows.update(tab.windowId, { width, height });
  return { success: true, width, height };
}

// ── Tab Management ───────────────────────────────────────────────

async function handleGetTabs() {
  const tabs = await chrome.tabs.query({ currentWindow: true });
  return {
    success: true,
    tabs: tabs.map(t => ({
      tabId: t.id,
      url: t.url,
      title: t.title,
      active: t.active,
    })),
  };
}

async function handleCreateTab() {
  const tab = await chrome.tabs.create({ active: true });
  return { success: true, tabId: tab.id, url: tab.url };
}

// ── Debugger Management ──────────────────────────────────────────

async function ensureDebuggerAttached(tabId) {
  if (debuggerAttached[tabId]) return;

  await chrome.debugger.attach({ tabId }, '1.3');
  debuggerAttached[tabId] = true;

  // Enable console and network domains for monitoring
  await chrome.debugger.sendCommand({ tabId }, 'Console.enable');
  await chrome.debugger.sendCommand({ tabId }, 'Network.enable');

  // Listen for console messages
  chrome.debugger.onEvent.addListener((source, method, params) => {
    if (source.tabId !== tabId) return;

    if (method === 'Console.messageAdded') {
      if (!consoleBuffer[tabId]) consoleBuffer[tabId] = [];
      consoleBuffer[tabId].push({
        level: params.message.level,
        text: params.message.text,
        url: params.message.url,
        timestamp: Date.now(),
      });
    }

    if (method === 'Network.requestWillBeSent') {
      if (!networkBuffer[tabId]) networkBuffer[tabId] = [];
      networkBuffer[tabId].push({
        url: params.request.url,
        method: params.request.method,
        type: params.type,
        timestamp: Date.now(),
      });
    }
  });
}

// Clean up debugger on tab close
chrome.tabs.onRemoved.addListener((tabId) => {
  if (debuggerAttached[tabId]) {
    try {
      chrome.debugger.detach({ tabId });
    } catch (e) { /* tab already gone */ }
    delete debuggerAttached[tabId];
    delete consoleBuffer[tabId];
    delete networkBuffer[tabId];
  }
});

// ── Popup Communication ──────────────────────────────────────────

function updatePopupStatus(connected) {
  chrome.runtime.sendMessage({
    type: 'status_update',
    connected,
    agentUrl: AGENT_WS_URL,
  }).catch(() => { /* popup not open */ });
}

// Listen for popup messages
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'get_status') {
    sendResponse({ connected: agentConnected, agentUrl: AGENT_WS_URL });
  } else if (message.type === 'connect') {
    connectToAgent();
    sendResponse({ ok: true });
  } else if (message.type === 'disconnect') {
    disconnectFromAgent();
    sendResponse({ ok: true });
  }
  return true;
});

// ── Content Script Functions (injected via executeScript) ────────

function buildAccessibilityTree() {
  let refCounter = 0;
  const INTERACTIVE_TAGS = new Set([
    'A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'DETAILS', 'SUMMARY',
  ]);
  const ROLE_MAP = {
    A: 'link', BUTTON: 'button', INPUT: 'textbox', SELECT: 'combobox',
    TEXTAREA: 'textarea', H1: 'heading', H2: 'heading', H3: 'heading',
    H4: 'heading', H5: 'heading', H6: 'heading', IMG: 'img',
    NAV: 'navigation', MAIN: 'main', HEADER: 'banner', FOOTER: 'contentinfo',
    FORM: 'form', TABLE: 'table', UL: 'list', OL: 'list', LI: 'listitem',
    P: 'paragraph', DIV: 'generic', SPAN: 'generic', SECTION: 'region',
  };

  function processNode(element, depth = 0) {
    if (depth > 15) return null;
    if (!element || element.nodeType !== 1) return null;
    if (element.offsetParent === null && element.tagName !== 'BODY' && element.tagName !== 'HTML') {
      // Hidden element
    }

    const refId = `ref_${++refCounter}`;
    const role = element.getAttribute('role') || ROLE_MAP[element.tagName] || 'generic';
    const name = element.getAttribute('aria-label')
               || element.getAttribute('alt')
               || element.getAttribute('title')
               || (element.tagName === 'INPUT' ? element.getAttribute('placeholder') || '' : '')
               || element.textContent?.trim().substring(0, 100) || '';
    const value = element.value || '';
    const interactive = INTERACTIVE_TAGS.has(element.tagName)
                      || element.getAttribute('role') === 'button'
                      || element.getAttribute('tabindex') !== null;

    const rect = element.getBoundingClientRect();
    const bounds = {
      x: Math.round(rect.x),
      y: Math.round(rect.y),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
    };

    const children = [];
    for (const child of element.children) {
      const childNode = processNode(child, depth + 1);
      if (childNode) children.push(childNode);
    }

    return { ref_id: refId, role, name, value, interactive, visible: true, bounds, children };
  }

  return processNode(document.body);
}

function findElementsInPage(query) {
  const queryLower = query.toLowerCase();
  const results = [];

  function scoreElement(el) {
    const text = (el.textContent || '').toLowerCase().trim();
    const ariaLabel = (el.getAttribute('aria-label') || '').toLowerCase();
    const role = (el.getAttribute('role') || el.tagName || '').toLowerCase();
    let score = 0;
    if (text.includes(queryLower)) score += 5;
    if (ariaLabel.includes(queryLower)) score += 8;
    if (role.includes(queryLower)) score += 4;
    return score;
  }

  const allElements = document.querySelectorAll('*');
  for (const el of allElements) {
    const score = scoreElement(el);
    if (score > 0) {
      const rect = el.getBoundingClientRect();
      results.push({
        score,
        ref_id: `found_${results.length}`,
        role: el.getAttribute('role') || el.tagName.toLowerCase(),
        name: el.getAttribute('aria-label') || el.textContent?.trim().substring(0, 100) || '',
        bounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
        interactive: el.tagName === 'BUTTON' || el.tagName === 'A' || el.tagName === 'INPUT',
      });
    }
  }

  results.sort((a, b) => b.score - a.score);
  return results.slice(0, 20);
}

function setFormValueInPage(refId, value) {
  // Find element by data attribute or index
  const idx = parseInt(refId.replace('ref_', ''), 10);
  const inputs = document.querySelectorAll('input, textarea, select, [contenteditable]');
  if (idx > 0 && idx <= inputs.length) {
    const el = inputs[idx - 1];
    el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return { success: true, value: el.value };
  }
  return { success: false, error: `Element ${refId} not found` };
}

console.log('[Cowork Bridge] Service worker loaded');
