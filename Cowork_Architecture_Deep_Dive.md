# Cowork Mode — Full Architecture & Workflow Deep Dive

## 1. Foundation Layer

### Orchestration
- **Claude Agent SDK** — the framework for building custom agents on top of Claude
- **Claude Code** — Anthropic's CLI agentic coding tool that provides the core agent loop (tool use, code execution, file I/O, action chaining)
- **Model**: Claude Opus 4.6 (primary), with the ability to delegate to Sonnet or Haiku for sub-tasks via the Task tool

### Execution Environment
- **Lightweight Linux VM** (Ubuntu 22) running locally on the user's machine
- Sandboxed — the VM's internal filesystem resets between sessions
- A **workspace folder** is mounted from the user's actual computer, providing persistent file access
- Pre-installed: Bash, Python, Node.js, npm, pip, LibreOffice, Poppler, pandoc, and more

### Session Structure
```
/sessions/<session-id>/                  # Ephemeral VM workspace (resets)
/sessions/<session-id>/mnt/<folder>/     # Mounted user workspace (persists)
/sessions/<session-id>/mnt/.skills/      # Skills plugin directory
/sessions/<session-id>/mnt/uploads/      # User-uploaded files
```

---

## 2. Tool System Architecture

Cowork exposes tools to the model in categories. Each tool call is a structured function invocation with JSON parameters.

### 2.1 File & Code Tools (Built-in)

| Tool | Purpose |
|------|---------|
| `Bash` | Execute shell commands (git, npm, docker, python, etc.) |
| `Read` | Read files (text, images, PDFs, notebooks) |
| `Write` | Create new files |
| `Edit` | Modify existing files via exact string replacement |
| `Glob` | Fast file pattern matching (e.g., `**/*.tsx`) |
| `Grep` | Content search using ripgrep regex |
| `NotebookEdit` | Edit Jupyter notebook cells |

### 2.2 Web Tools

| Tool | Purpose |
|------|---------|
| `WebFetch` | Fetch a URL, convert HTML to markdown, process with AI |
| `WebSearch` | Search the web for up-to-date information |

### 2.3 Agent/Task Tools

| Tool | Purpose |
|------|---------|
| `Task` | Spawn sub-agents (specialized autonomous agents for complex tasks) |
| `TaskOutput` | Retrieve output from running/completed background tasks |
| `TaskStop` | Stop a running background task |

**Sub-agent Types Available via Task:**
- `Bash` — Command execution specialist
- `Explore` — Fast codebase exploration (quick/medium/thorough)
- `Plan` — Software architecture planning
- `general-purpose` — Multi-step research and execution
- `statusline-setup` — Configure Claude Code status line
- `claude-code-guide` — Answer questions about Claude Code, Agent SDK, API

Each sub-agent runs independently, can use a different model (sonnet, opus, haiku), and optionally runs in an isolated git worktree.

### 2.4 User Interaction Tools

| Tool | Purpose |
|------|---------|
| `AskUserQuestion` | Present 1-4 multiple-choice questions to the user |
| `TodoWrite` | Manage a visible task progress widget |
| `EnterPlanMode` / `ExitPlanMode` | Structured planning workflow with user approval |
| `EnterWorktree` | Create an isolated git worktree for a session |

### 2.5 Browser Automation (Claude in Chrome)

Full browser control through an MCP server connected to a Chrome extension:

| Tool | Purpose |
|------|---------|
| `tabs_context_mcp` | Get info about current tab group |
| `tabs_create_mcp` | Create a new tab |
| `navigate` | Go to URL or back/forward |
| `read_page` | Get accessibility tree of page elements |
| `find` | Natural language element search |
| `computer` | Mouse/keyboard actions (click, type, scroll, screenshot, zoom, drag, hover) |
| `form_input` | Set values in form elements |
| `javascript_tool` | Execute JS in page context |
| `read_console_messages` | Read browser console output |
| `read_network_requests` | Read HTTP network requests |
| `get_page_text` | Extract raw text from page |
| `resize_window` | Resize browser window |
| `gif_creator` | Record browser actions as animated GIF |
| `upload_image` | Upload screenshots/images to page elements |
| `shortcuts_list` / `shortcuts_execute` | List and run browser shortcuts/workflows |
| `switch_browser` | Switch to a different Chrome browser |
| `update_plan` | Present a plan to user for browser action approval |

### 2.6 MCP (Model Context Protocol) Integrations

External service connectors discovered and connected at runtime:

| Tool | Purpose |
|------|---------|
| `search_mcp_registry` | Search for available connectors by keyword |
| `suggest_connectors` | Show connector suggestions with Connect buttons |

**Example MCP Servers (Zoho CRM shown here):**
- `ZohoCRM_Get_Records`, `ZohoCRM_Search_Records`, `ZohoCRM_Create_Records`
- `ZohoCRM_Update_Records`, `ZohoCRM_Delete_Records`, `ZohoCRM_Upsert_Records`
- `ZohoCRM_Add_Tags`, `ZohoCRM_Mass_Update_Records`, etc.

### 2.7 Scheduled Tasks

| Tool | Purpose |
|------|---------|
| `list_scheduled_tasks` | List all scheduled tasks with state |
| `create_scheduled_task` | Create new recurring or ad-hoc task |
| `update_scheduled_task` | Modify existing task (prompt, schedule, enabled) |

Tasks use cron expressions in local timezone and are stored as skill files.

### 2.8 File System Access Tools

| Tool | Purpose |
|------|---------|
| `request_cowork_directory` | Ask user to select a folder from their computer |
| `present_files` | Show files as interactive cards in chat |
| `allow_cowork_file_delete` | Request permission to delete files |

### 2.9 Plugin System

| Tool | Purpose |
|------|---------|
| `search_plugins` | Search for installable plugins |
| `suggest_plugin_install` | Show plugin installation banner |

Plugins are installable bundles of MCPs, skills, and tools, grouped into marketplaces.

---

## 3. Skills System Architecture

### 3.1 What is a Skill?

A skill is a folder containing a `SKILL.md` file with YAML frontmatter and markdown instructions. Skills provide domain-specific best practices and step-by-step guides for producing high-quality outputs.

### 3.2 Skill Directory Structure

```
/mnt/.skills/
├── .claude-plugin/
│   └── plugin.json          # Plugin metadata
├── manifest.json             # Skill registry with IDs, descriptions, enabled state
└── skills/
    ├── docx/
    │   ├── SKILL.md          # Main instructions
    │   └── LICENSE.txt
    ├── pdf/
    │   ├── SKILL.md
    │   ├── REFERENCE.md      # Advanced reference
    │   ├── FORMS.md           # Form filling guide
    │   └── LICENSE.txt
    ├── pptx/
    │   ├── SKILL.md
    │   ├── pptxgenjs.md       # Creation from scratch guide
    │   ├── editing.md         # Editing existing files guide
    │   └── LICENSE.txt
    ├── xlsx/
    │   ├── SKILL.md
    │   └── LICENSE.txt
    ├── remotion/
    │   ├── SKILL.md
    │   └── references/
    │       └── animations.md
    ├── schedule/
    │   └── SKILL.md
    └── skill-creator/
        ├── SKILL.md
        ├── LICENSE.txt
        ├── agents/
        │   ├── grader.md
        │   ├── comparator.md
        │   └── analyzer.md
        └── references/
            └── schemas.md
```

### 3.3 Progressive Disclosure Model

Skills use a three-level loading system:

1. **Metadata** (always in context, ~100 words) — name + description from manifest.json
2. **SKILL.md body** (loaded when skill triggers, <500 lines ideal) — core instructions
3. **Bundled resources** (loaded on demand, unlimited size) — scripts, references, assets

### 3.4 Skill Triggering

Skills appear in the system prompt's `<available_skills>` section. The model matches user requests to skills based on:
- **Keyword triggers** defined in the description (e.g., "Excel", "spreadsheet", ".xlsx")
- **Intent matching** — the model infers when a skill would help, even without explicit keywords
- **Skill tool invocation** — the `Skill` tool is called with the skill name, which loads the SKILL.md instructions

### 3.5 Skill Invocation Flow

```
User Request → Model matches to skill → Skill tool called → SKILL.md loaded
→ Model reads instructions → Model follows skill's workflow → Output produced
→ Verification/QA step → Final output delivered to user
```

### 3.6 Individual Skill Workflows

#### DOCX Skill
- **Read**: `pandoc` for text extraction, `unpack.py` for raw XML
- **Create new**: JavaScript with `docx-js` npm package → validate with `validate.py`
- **Edit existing**: Unpack → Edit XML directly → Repack (3-step process)
- **Tools/libs**: pandoc, docx (npm), LibreOffice, Poppler, custom scripts (unpack.py, pack.py, comment.py, accept_changes.py, soffice.py)

#### PDF Skill
- **Read/Extract**: `pypdf` for basic ops, `pdfplumber` for text/tables
- **Create**: `reportlab` (Canvas or Platypus)
- **Manipulate**: merge, split, rotate, watermark, encrypt/decrypt via pypdf/qpdf
- **OCR**: `pytesseract` + `pdf2image`
- **Forms**: Separate FORMS.md guide with `pdf-lib` (JS) or pypdf
- **Advanced**: REFERENCE.md for pypdfium2, JavaScript libraries

#### PPTX Skill
- **Read**: `markitdown` for text extraction, `thumbnail.py` for visual overview
- **Create from scratch**: `pptxgenjs` (npm) — see pptxgenjs.md
- **Edit existing**: Unpack → manipulate slides → edit content → clean → pack — see editing.md
- **Visual QA**: Convert to images via LibreOffice+pdftoppm → sub-agent visual inspection
- **Design**: Includes color palettes, typography guide, layout patterns, anti-patterns

#### XLSX Skill
- **Read/Analyze**: pandas for data analysis
- **Create/Edit**: openpyxl for formulas, formatting, Excel-specific features
- **Critical rule**: Always use Excel formulas, never hardcode calculated values
- **Formula recalc**: `scripts/recalc.py` (uses LibreOffice) — mandatory after creating/editing
- **Verification**: JSON output from recalc.py with error details, must fix all formula errors

#### Remotion Skill
- **Setup**: `npm init video` or add `remotion @remotion/cli` to existing project
- **Create**: React components with Remotion hooks (`useCurrentFrame`, `interpolate`, `spring`)
- **Compose**: `<Sequence>`, `<Audio>`, `<Video>` for timeline control
- **Render**: `npx remotion render` for video, `npx remotion still` for frames

#### Schedule Skill
- **Workflow**: Analyze session → Draft self-contained prompt → Choose taskId → Determine schedule
- **Output**: Calls `create_scheduled_task` tool with cron expression
- **Cron**: Local timezone, standard 5-field format

#### Skill Creator (Meta-Skill)
- **Full workflow**: Capture intent → Interview/research → Write SKILL.md → Create test cases → Run evals (with-skill vs baseline) → Grade → Aggregate benchmarks → Launch viewer → Read feedback → Iterate
- **Progressive Disclosure**: Metadata → SKILL.md → Bundled resources
- **Description optimization**: Generate trigger eval queries → User review via HTML template → Automated optimization loop with `run_loop` script
- **Sub-agents used**: grader, comparator, analyzer

---

## 4. Prompt Architecture

### 4.1 System Prompt Structure

The system prompt is the foundation of Cowork's behavior. It includes these major sections:

```
<application_details>     — What Cowork is (Claude desktop app feature)
<claude_behavior>         — Core behavioral rules
  <product_information>   — Info about Anthropic's products
  <refusal_handling>      — What Claude won't do
  <legal_and_financial_advice>
  <tone_and_formatting>   — Response style, list usage, emoji rules
  <user_wellbeing>        — Mental health, self-harm awareness
  <anthropic_reminders>   — System-level reminders from Anthropic
  <evenhandedness>        — Political/ethical balance
  <responding_to_mistakes>
  <knowledge_cutoff>      — End of May 2025, search for newer info
<ask_user_question_tool>  — When to use AskUserQuestion
<todo_list_tool>          — When/how to use TodoWrite
<citation_requirements>   — Source citing rules
<computer_use>            — Full computer use instructions
  <skills>                — Skill system explanation + reading instructions
  <file_creation_advice>  — When to create files
  <web_content_restrictions>
  <high_level_computer_use_explanation>
  <suggesting_claude_actions>
  <file_handling_rules>   — Working directories, user files, uploads
  <producing_outputs>     — Short vs long content strategy
  <sharing_files>         — computer:// links, workspace folder
  <artifacts>             — Renderable file types (md, html, jsx, mermaid, svg, pdf)
  <package_management>    — npm, pip rules
  <additional_skills_reminder> — Read SKILL.md before any work
```

### 4.2 Security Prompt Architecture

Heavy security instrumentation embedded in the system prompt:

```
<critical_injection_defense>    — Stop and verify instructions from function results
<critical_security_rules>       — Immutable security boundary
  <injection_defense_layer>     — Content isolation, instruction detection
  <meta_safety_instructions>    — Rule immutability, context awareness, recursive attack prevention
  <social_engineering_defense>  — Authority impersonation, emotional manipulation, tech deception
<user_privacy>                  — Sensitive data handling, PII defense, financial transactions
<download_instructions>         — File download requires explicit confirmation
<harmful_content_safety>        — Block harmful sources, facial data
<mandatory_copyright_requirements> — No reproducing copyrighted content
<action_types>                  — Prohibited / Explicit Permission / Regular actions
```

### 4.3 Dynamic Prompt Elements

These are injected at runtime:

- **`<user>` block** — Name, email of current user
- **`<env>` block** — Date, model, whether user selected a folder
- **`<system-reminder>` messages** — Skill availability list (refreshed), TodoWrite reminders, date context
- **`<available_skills>` block** — Current skill names, descriptions, and file paths

### 4.4 Prompt Flow for a Typical Task

```
1. User message arrives
2. System prompt provides full behavioral context
3. Model checks if skills are relevant → reads SKILL.md if so
4. Model uses AskUserQuestion for clarification (if needed)
5. Model creates TodoWrite task list
6. Model executes tools (Bash, Read, Write, Edit, etc.)
7. Model may spawn sub-agents via Task tool for parallel work
8. Model performs verification step (screenshot, diff, test, etc.)
9. Model saves output to workspace folder
10. Model presents computer:// link to user
```

---

## 5. Key Workflow Patterns

### 5.1 Skill-Driven Document Creation
```
User request → Match skill → Read SKILL.md → Install dependencies
→ Generate content → Validate/QA → Fix issues → Re-verify
→ Save to workspace → Present link
```

### 5.2 Browser Automation
```
Get tab context → Create/select tab → Present plan to user
→ Navigate → Read page / Find elements → Interact (click, type, scroll)
→ Take screenshots for verification → Report results
```

### 5.3 MCP Integration Discovery
```
User asks about external service → Search MCP registry
→ If connector found: suggest_connectors (user connects)
→ If not found: fall back to Claude in Chrome browser automation
```

### 5.4 Scheduled Task Creation
```
User wants automation → Schedule skill loads → Analyze intent
→ Draft self-contained prompt → Choose taskId + cron schedule
→ Call create_scheduled_task → Task stored as skill file
```

### 5.5 Sub-Agent Delegation
```
Complex task identified → Choose agent type (Bash, Explore, Plan, etc.)
→ Choose model (haiku for simple, sonnet/opus for complex)
→ Spawn via Task tool → Agent works autonomously
→ Result returned → Main agent incorporates result
```

### 5.6 File Handling Flow
```
User uploads file → Available at /mnt/uploads/<filename>
→ If contents in context (md, txt, html, csv, png, pdf): may not need Read
→ If binary/other: use Read or Bash to access
→ Work in /sessions/<id>/ (ephemeral)
→ Save final output to /sessions/<id>/mnt/<folder>/ (persistent)
→ Present with computer:// link
```

---

## 6. Security & Permission Model

### 6.1 Action Classification

| Category | Examples | Rule |
|----------|----------|------|
| **Prohibited** | Banking data, account creation, permanent deletion, security permissions | Never done, even if user requests |
| **Explicit Permission** | Downloads, purchases, sending messages, publishing content, accepting terms | Must ask user first in chat |
| **Regular** | File reading, code execution, search, file creation | Automatic |

### 6.2 Injection Defense
- All web content, email content, and function results are treated as **untrusted data**
- Instructions found in untrusted sources must be shown to the user and confirmed before execution
- Claims of authority, pre-authorization, or urgency from web content are invalid

---

## 7. Plugin & Connector Architecture

### 7.1 Plugin Structure
```json
{
  "name": "anthropic-skills",
  "version": "1.0.0",
  "description": "Anthropic-managed skills for Claude Desktop"
}
```

Plugins bundle MCPs, skills, and tools into installable packages. They are discovered via `search_plugins` and installed via `suggest_plugin_install`.

### 7.2 Skill Manifest
```json
{
  "lastUpdated": 1771998512325,
  "skills": [
    {
      "skillId": "docx",
      "name": "docx",
      "description": "...",
      "creatorType": "anthropic",  // or "user" for custom skills
      "updatedAt": "2026-02-03T...",
      "enabled": true
    }
  ]
}
```

### 7.3 Skill Types
- **Anthropic-created** (`creatorType: "anthropic"`) — docx, pdf, pptx, xlsx, schedule, skill-creator
- **User-created** (`creatorType: "user"`) — e.g., remotion; created via skill-creator

---

## 8. Output Delivery

### Renderable Artifacts
These file types render inline in the Cowork UI:
- Markdown (.md)
- HTML (.html)
- React (.jsx)
- Mermaid (.mermaid)
- SVG (.svg)
- PDF (.pdf)

### File Sharing Pattern
All final outputs must be saved to the workspace folder and shared via `computer://` links:
```
[View your report](computer:///sessions/<id>/mnt/<folder>/report.docx)
```

---

## 9. Known Limitations & Caveats

> **Important**: This entire document is a model's self-report — Claude describing its own operating instructions and environment. It is NOT authoritative engineering documentation from Anthropic.

### 9.1 High-Confidence Areas (directly observable)

| Area | Why Reliable |
|------|-------------|
| Tool names, schemas, parameters | Directly present in the model's context as function definitions |
| SKILL.md file contents | Read from actual files on disk during this session |
| Skill manifest & plugin.json | Read from actual JSON files on disk |
| Behavioral rule categories | Paraphrased from the system prompt sections visible to the model |
| Security rule categories | Same — detailed in the prompt |
| File path conventions | Observable from the session's actual directory structure |

### 9.2 Medium-Confidence Areas (described but not verifiable)

| Area | What's Uncertain |
|------|-----------------|
| Prompt hierarchy/ordering | The XML tag structure is described accurately, but exact ordering, whether sections are conditionally included, and whether there's preprocessing is unknown |
| Skill triggering logic | Described as keyword + intent matching, but there may be classifiers, confidence thresholds, or A/B testing involved |
| System-reminder injection | We saw a `<system-reminder>` appear mid-conversation, but what triggers it, how often it refreshes, and what controls its content is opaque |
| Sub-agent architecture | The Task tool and agent types are documented, but how sub-agents are actually spawned (separate API calls? separate sessions? shared context?) is implementation detail |
| Action classification boundaries | The prohibited/permission/regular categories are listed, but edge cases are resolved by model judgment, which is probabilistic |

### 9.3 Low-Confidence Areas (mostly inferred or unknown)

| Area | What's Unknown |
|------|---------------|
| **Orchestration runtime** | "Claude Code + Agent SDK" is named in the prompt, but the actual execution pipeline — how sessions are created, how tool calls are dispatched to the VM, how results return, error handling, retries, load balancing — is entirely opaque |
| **VM/container technology** | "Lightweight Linux VM (Ubuntu 22)" is stated, but the containerization tech (Docker, Firecracker, gVisor?), networking, resource limits, and provisioning pipeline are unknown |
| **MCP protocol implementation** | Tool schemas are visible, but the actual protocol (WebSocket? HTTP? gRPC?), authentication, token management, connection lifecycle, and error handling are not documented in the prompt |
| **Browser automation bridge** | Chrome extension MCP tools are documented by schema, but the communication channel, screenshot transmission, latency, and failure modes are invisible |
| **Prompt versioning & A/B testing** | Skills have `updatedAt` timestamps and the manifest has `lastUpdated`, suggesting the prompt changes frequently. Whether different users see different prompt variants is unknown |
| **Additional safety systems** | The prompt describes behavioral safety rules, but whether there are additional runtime systems (output classifiers, toxicity filters, monitoring dashboards) beyond the prompt is unknown |
| **Model selection logic** | The prompt mentions haiku/sonnet/opus for sub-agents, but how the primary model for a session is chosen (user setting? auto-selection? plan-based?) is not described |

### 9.4 Things That May Be Missing Entirely

- **Telemetry & logging** — How Anthropic monitors Cowork sessions, what's logged, retention policies
- **Rate limiting** — Per-tool, per-session, or per-user limits on tool calls, API calls, or compute
- **Cost management** — How token usage is tracked, billed, or capped
- **Session lifecycle** — How sessions start, timeout, resume, and terminate
- **Multi-user / team features** — Whether Cowork supports shared workspaces, team settings, or role-based access
- **Offline behavior** — What happens when network connectivity is lost mid-session
- **Update mechanism** — How skills, plugins, and the prompt itself are updated across sessions
- **Fallback behavior** — What happens when tools fail, the VM crashes, or MCP servers are unreachable

### 9.5 Potential Inaccuracies in Specific Claims

1. **"The VM's internal filesystem resets between sessions"** — This is stated in the prompt, but "reset" could mean full re-provisioning, snapshot restore, or just cleanup. The mechanism matters for security analysis but is unknown.

2. **"Skills use a three-level progressive disclosure model"** — This is from the skill-creator SKILL.md, which describes it as a design pattern. Whether the runtime actually enforces these levels or it's just a convention is unclear.

3. **"40+ tools"** — The document counts tools across categories, but some MCP tools may be dynamically loaded/unloaded based on which connectors the user has enabled. The number is session-dependent.

4. **"Security rules are immutable"** — The prompt says they are, but from a technical standpoint, the system prompt is text provided to an LLM. Whether these rules can be bypassed through adversarial techniques is an active research area, not a settled fact.

5. **"Runtime context injection"** — The document describes `<user>`, `<env>`, and `<system-reminder>` as "injected at runtime." This is what it looks like from inside, but the actual mechanism (template substitution before API call? separate system messages? injected into the user turn?) is an implementation detail.

### 9.6 How to Use These Documents

- **For learning about prompt engineering**: The behavioral sections, XML hierarchy, and design patterns (Section 5 of the System Prompt doc) are solid reference material
- **For understanding Cowork's capabilities**: The tool catalog and skill workflows are accurate and comprehensive
- **For security analysis**: The security rules describe the *intended* behavior; actual robustness requires adversarial testing
- **For building similar systems**: Use as a reference architecture, but don't treat implementation details as confirmed

---

*Document generated from live Cowork session analysis on February 25, 2026. This is a model self-report with known limitations — see Section 9 above.*
