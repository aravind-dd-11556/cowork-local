# Security Audit Report — Sprint 46: Chrome Extension Bridge

**Date:** 2026-03-04
**Scope:** Full codebase audit with focus on Sprint 46 (Chrome Extension Bridge)
**Regression Status:** 4,960 tests passed, 0 failures

---

## Executive Summary

The audit identified **37 vulnerabilities** across three audit domains. The Chrome Extension Bridge introduces the highest-risk attack surface due to WebSocket-based inter-process communication with a real browser. The core agent code demonstrates strong foundational security practices (YAML safe loading, credential masking, prompt injection detection, subprocess hardening), but the new bridge code lacks critical hardening.

| Severity | Python Bridge | Chrome Extension JS | Core Agent | Total |
|----------|:---:|:---:|:---:|:---:|
| CRITICAL | 2 | 3 | 2 | **7** |
| HIGH | 4 | 7 | 4 | **15** |
| MEDIUM | 3 | 4 | 5 | **12** |
| LOW | 1 | 3 | 4 | **8** |
| **Total** | **10** | **17** | **15** | **42** |

**Overall Risk: MODERATE-HIGH** — Suitable for localhost development only. Not production-ready without remediation.

---

## CRITICAL Vulnerabilities (7)

### C-1: Unencrypted WebSocket (No TLS)
- **Files:** `chrome_ws_client.py:30,71` / `background.js:15`
- **Issue:** Default `ws://` (plaintext). All JSON-RPC messages—including JS code, form values, screenshots, and page content—transmitted unencrypted.
- **Impact:** Man-in-the-middle attacks, credential interception, data exfiltration.
- **Fix:** Enforce `wss://` for non-localhost; validate URL scheme on connect.

### C-2: Unbounded Pending Requests (Memory Exhaustion)
- **File:** `chrome_ws_client.py:50,153`
- **Issue:** `_pending` dict grows without limit. If responses never arrive, futures accumulate indefinitely.
- **Impact:** Memory exhaustion DoS.
- **Fix:** Add `MAX_PENDING_REQUESTS = 1000` cap; reject new calls when full.

### C-3: Unsafe `eval()` in Chrome Extension
- **File:** `background.js:263-276`
- **Issue:** `handleExecuteScript` uses `eval(jsCode)` to run arbitrary JavaScript on any tab.
- **Impact:** Remote code execution, complete browser compromise.
- **Fix:** Replace with `new Function()` + input validation, or use sandboxed iframe.

### C-4: No WebSocket Authentication
- **File:** `background.js:15,19-68`
- **Issue:** Extension connects to `ws://localhost:9222` with zero authentication. Any local process can impersonate the agent.
- **Impact:** Command injection, unauthorized browser control.
- **Fix:** Implement token-based auth (shared secret on connect handshake).

### C-5: Overly Broad `<all_urls>` Permission
- **File:** `manifest.json:12-15`
- **Issue:** Extension requests `<all_urls>` host permission — access to every website.
- **Impact:** One vulnerability compromises all user browsing (banking, email, healthcare).
- **Fix:** Use specific host permissions; use `optional_permissions` for broader access.

### C-6: JS Execution Callback Not Validated
- **File:** `browser_session.py:145,170,187`
- **Issue:** `on_js_execute` callback registered without validation. If hijacked, enables arbitrary JS execution.
- **Impact:** RCE in browser context, credential theft.
- **Fix:** Validate callbacks are from trusted sources; add audit logging.

### C-7: MCP Environment Variable Injection
- **File:** `connector_auth.py:815-829`
- **Issue:** `build_mcp_env()` sets arbitrary env vars from `cfg.token_env_var`. Could set `LD_PRELOAD`, `PATH`, `PYTHONPATH`.
- **Impact:** Library hijacking, arbitrary code execution via subprocess.
- **Fix:** Whitelist allowed env var names; reject dangerous names.

---

## HIGH Vulnerabilities (15)

### H-1: Missing JSON-RPC Message Validation
- **File:** `chrome_ws_client.py:181-207`
- **Issue:** `_listen_loop()` accepts any JSON without validating structure, size, or types.
- **Fix:** Add message size limit (10MB), validate JSON-RPC 2.0 structure, type-check `id` field.

### H-2: Information Leakage in Error Messages
- **Files:** `chrome_ws_client.py:89,204` / `chrome_bridge.py:134,190,200,210`
- **Issue:** Full exception details (paths, stack traces, server errors) exposed in logs and return values.
- **Fix:** Use generic error messages; log details only at DEBUG level.

### H-3: Unbounded Reconnect Loop
- **File:** `chrome_ws_client.py:235-252`
- **Issue:** Infinite reconnection with no max attempts. Multiple `_reconnect()` tasks can run concurrently.
- **Fix:** Add `max_reconnect_attempts=10`; use a lock to prevent concurrent reconnects.

### H-4: TOCTOU Race Condition
- **File:** `chrome_ws_client.py:138,156`
- **Issue:** Connection checked at line 138, used at line 156. Connection can drop between check and send.
- **Fix:** Wrap actual send in try-except; handle `ConnectionError` at the send site.

### H-5: Form Input Index Manipulation
- **File:** `background.js:495-507`
- **Issue:** `parseInt(refId.replace('ref_', ''))` with no bounds or format validation. Could access wrong elements.
- **Fix:** Validate with regex `/^ref_\d+$/`; add bounds checking.

### H-6: Debugger API Over-Permission
- **File:** `background.js:333-367`
- **Issue:** Enables Console and Network CDP domains permanently. Full CDP access on all debugger-attached tabs.
- **Fix:** Enable domains only when needed; detach debugger when idle.

### H-7: Screenshot/Page Text Captures Sensitive Data
- **File:** `background.js:133-139,280-287`
- **Issue:** Screenshots and page text captured without filtering. Captures banking, email, healthcare content.
- **Fix:** Detect sensitive page patterns; require user consent for capture.

### H-8: Console Message Buffer Overflow
- **File:** `background.js:14,348-355`
- **Issue:** `consoleBuffer` grows without limit per tab.
- **Fix:** Implement circular buffer with max 1000 entries per tab.

### H-9: Network Request Buffer Overflow
- **File:** `background.js:14,357-365`
- **Issue:** `networkBuffer` grows without limit per tab.
- **Fix:** Same as H-8 — circular buffer with size cap.

### H-10: Keyboard Input Injection
- **File:** `background.js:199-210`
- **Issue:** `type` action dispatches key events without filtering control characters or dangerous shortcuts.
- **Fix:** Filter control characters; blacklist dangerous key combinations.

### H-11: Path Depth Check Bypass (Symlinks)
- **File:** `tools/write.py:35-57`
- **Issue:** Path depth check doesn't `resolve()` symlinks before validation.
- **Fix:** Use `Path(file_path).resolve()` before depth check.

### H-12: Tool Generator Sandbox Escape Risk
- **File:** `tool_generator.py:260-271`
- **Issue:** `_run_in_sandbox()` accesses `__builtins__` directly without proper isolation.
- **Fix:** Use `compile()` with restricted globals/locals; never expose `exec` in restricted namespace.

### H-13: Connector Auth Env Variable Injection
- **File:** `connector_auth.py:815-829`
- **Issue:** Arbitrary env var names from config passed to subprocess without validation.
- **Fix:** Whitelist safe env var names (e.g., `SLACK_BOT_TOKEN`, `GITHUB_TOKEN`).

### H-14: No Input Size Limits on Browser Callbacks
- **File:** `browser_session.py:160-191`
- **Issue:** Callbacks accept arbitrarily large inputs (text, coordinates, code).
- **Fix:** Add size limits: 1MB for JS code, 10KB for form values, bounds for coordinates.

### H-15: Unvalidated Parameters in Bridge Methods
- **File:** `chrome_bridge.py:121-253`
- **Issue:** `tab_id`, `query`, `code` parameters passed to RPC without validation.
- **Fix:** Validate types, ranges, and sizes before RPC calls.

---

## MEDIUM Vulnerabilities (12)

| # | Issue | File | Fix |
|---|-------|------|-----|
| M-1 | No origin/protocol validation on WS connect | `chrome_ws_client.py:71` | Add origin header; enforce localhost for `ws://` |
| M-2 | Sync callbacks silently return None in running loop | `chrome_bridge.py:259-287` | Add warning logs; document async-only usage |
| M-3 | No CSP in extension manifest | `manifest.json` | Add `content_security_policy`; remove inline onclick |
| M-4 | refId string parsing vulnerability | `background.js:497` | Validate with regex before parseInt |
| M-5 | No timeout on CDP commands | `background.js:178-259` | Use `Promise.race()` with 30s timeout |
| M-6 | Accessibility tree exposes sensitive content | `background.js:407-458` | Redact password fields; filter sensitive URLs |
| M-7 | Hardcoded localhost URLs | `default_config.yaml:11,91` | Make env-variable configurable; warn on HTTP |
| M-8 | Session cleanup not automatic | `api.py:100-109` | Add background cleanup task |
| M-9 | WebSocket message deserialization unvalidated | `api.py:282-293` | Add Pydantic validation; limit JSON size |
| M-10 | ReDoS risk in credential detector | `credential_detector.py:48-127` | Add regex timeout; anchor patterns |
| M-11 | No CORS configuration on API | `api.py` | Add CORSMiddleware with origin whitelist |
| M-12 | Content script missing error handling | `content-script.js:10-20` | Add try-catch; validate message schema |

---

## LOW Vulnerabilities (8)

| # | Issue | File |
|---|-------|------|
| L-1 | Request ID counter never resets | `chrome_ws_client.py:49` |
| L-2 | Hardcoded WS URL in extension | `background.js:15` |
| L-3 | No logging/audit trail in extension | `background.js` |
| L-4 | No rate limiting in extension | `background.js` |
| L-5 | Console/network buffers not cleared between tool calls | `browser_session.py:100-102` |
| L-6 | Predictable group IDs (8-char UUID hex) | `browser_session.py:206` |
| L-7 | Health check endpoint unauthenticated | `api.py` |
| L-8 | No rate limiting on API endpoints | `api.py` |

---

## Positive Security Controls Found

The codebase has several strong security practices already in place:

1. **YAML Safe Loading** — `yaml.safe_load()` used everywhere (no unsafe deserialization)
2. **Credential Masking** — Comprehensive credential detection and masking system
3. **Prompt Injection Detection** — Multi-pattern detection for injection attacks
4. **Subprocess Hardening** — MCP client validates commands against whitelist; `shell=False`
5. **Code Generation Sandbox** — Generated tools validated for dangerous imports/calls
6. **Resource Limits** — Sandboxed executor enforces timeouts and memory limits
7. **Session Isolation** — Each API session gets independent agent instance
8. **File Write Protection** — Absolute path requirement and depth limiting
9. **Path Traversal Prevention** — UUID sanitization in connector auth
10. **Token Length Validation** — Environment variable length checks

---

## Remediation Priority

### Immediate (Before any production use)
1. Implement WebSocket authentication (shared secret/token)
2. Replace `eval()` with safer JS execution
3. Restrict `<all_urls>` to specific permissions
4. Whitelist env var names in connector auth
5. Add pending request limit and message size validation

### Short-term (Next sprint)
6. Enforce WSS for non-localhost connections
7. Fix path traversal (resolve symlinks)
8. Add input size limits on all bridge methods
9. Implement buffer caps for console/network data
10. Fix TOCTOU race conditions with proper error handling

### Medium-term (2-3 sprints)
11. Add CORS middleware to API
12. Implement rate limiting across all endpoints
13. Add CSP to extension manifest
14. Implement automatic session cleanup
15. Add audit logging throughout

---

## Conclusion

The Sprint 46 Chrome Extension Bridge introduces a significant attack surface by connecting the Python agent to a real browser via WebSocket. The core agent code has solid security foundations, but the bridge layer needs hardening before any non-development use. The 7 critical issues should be addressed before the extension is used outside of localhost development environments.
