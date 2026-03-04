---
name: schedule
description: "Schedule and automate recurring tasks. Create Scheduled tasks that run on demand or on intervals.\n  MANDATORY TRIGGERS: schedule, cron, recurring, automated task, interval, timer"
---

You are creating a reusable scheduled task. Follow these steps:

## 1. Analyze the Session

Review the session history to identify the core task. Distill it into a single, repeatable objective.

## 2. Draft a Prompt

The prompt will be used for future autonomous runs — it must be entirely self-contained. Future runs will NOT have access to this session.

Include in the description:
- A clear objective statement (what to accomplish)
- Specific steps to execute
- Any relevant file paths, URLs, repositories, or tool names
- Expected output or success criteria
- Any constraints or preferences the user expressed

Write in second-person imperative ("Check the inbox...", "Run the test suite...").

## 3. Choose a taskId

Pick a short, descriptive name in kebab-case (e.g., "daily-inbox-summary", "weekly-dep-audit").

## 4. Determine Scheduling

- **Clearly one-off** (e.g., "refactor this function") — omit cron expression
- **Clearly recurring** (e.g., "check inbox every morning") — include cron expression
- **Ambiguous** — propose a schedule and ask user to confirm

**IMPORTANT: Cron expressions run in the user's local timezone, NOT UTC.**

### Cron Expression Format

Format: `minute hour dayOfMonth month dayOfWeek`

### Common Patterns
```
0 9 * * *       → Every day at 9:00 AM
0 9 * * 1-5     → Weekdays at 9:00 AM
30 8 * * 1      → Every Monday at 8:30 AM
0 0 1 * *       → First of every month at midnight
0 */2 * * *     → Every 2 hours
0 9,17 * * *    → At 9 AM and 5 PM daily
```

### Example — Daily Standup
```
taskId: daily-standup
description: Generate daily standup summary from git commits
cronExpression: 0 9 * * 1-5
prompt: |
  Check the git log for commits made yesterday.
  Summarize: Done yesterday, Planned today, Blockers.
```

### Example — Weekly Report
```
taskId: weekly-report
description: Create weekly progress report
cronExpression: 0 17 * * 5
prompt: |
  Analyze git commits from the past week.
  Create a markdown report with key accomplishments,
  files changed, and lines added/removed.
```

## 5. Create the Task

Finally, call the `create_scheduled_task` tool with your taskId, prompt, description, and optional cronExpression.

## Managing Tasks

- **List**: `list_scheduled_tasks`
- **Update**: `update_scheduled_task` to change schedule or prompt
- **Pause**: Set `enabled: false`
- **Delete**: `delete_scheduled_task`
- **Run now**: `run_scheduled_task`
