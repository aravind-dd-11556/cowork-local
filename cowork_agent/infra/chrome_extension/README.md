# Open-Cowork Chrome Extension Bridge

Chrome extension that bridges the Open-Cowork agent to a real Chrome browser via WebSocket.

## Installation

1. Open Chrome and navigate to `chrome://extensions`
2. Enable "Developer mode" (toggle in top-right corner)
3. Click "Load unpacked" and select this directory
4. The extension icon will appear in the toolbar

## Usage

1. Start the Open-Cowork agent with `chrome_bridge.enabled: true`
2. Click the extension icon in Chrome toolbar
3. Click "Connect to Agent"
4. The agent can now control Chrome — navigating pages, clicking elements, taking screenshots, etc.

## Architecture

```
Agent (Python)  ←→  WebSocket  ←→  Extension (JS)  ←→  Chrome APIs
```

The extension acts as a WebSocket MCP server that:
- Receives JSON-RPC 2.0 calls from the agent
- Translates them into Chrome API calls (tabs, debugger, scripting)
- Returns results back to the agent

## Supported Operations

| Operation | Chrome API |
|-----------|-----------|
| Navigate | `chrome.tabs.update()` |
| Screenshot | `chrome.tabs.captureVisibleTab()` |
| Click/Type/Scroll | CDP `Input.dispatch*Event` via `chrome.debugger` |
| Read Page | Content script DOM traversal |
| Execute JS | `chrome.scripting.executeScript()` |
| Get Page Text | `document.body.innerText` |
| Console Messages | CDP `Console.messageAdded` |
| Network Requests | CDP `Network.requestWillBeSent` |
| Resize Window | `chrome.windows.update()` |

## Security

- WebSocket only accepts connections from localhost
- User must manually click "Connect" to enable bridge
- CDP debugger requires explicit permission per tab
