---
name: schedule
description: "Create scheduled tasks that can be run on demand or automatically on a recurring interval."
---

# Schedule Skill

MANDATORY TRIGGERS: schedule, scheduled task, recurring, interval, cron, automate, timer, periodic

## Overview

Create tasks that run automatically on a schedule or can be triggered manually.

## Creating a Scheduled Task

Use the `create_scheduled_task` tool with these parameters:

- **taskId**: Kebab-case identifier (e.g., `daily-report`, `check-inbox`)
- **prompt**: Complete instructions for what the task should do
- **description**: One-line summary
- **cronExpression**: Optional 5-field cron for automatic scheduling

## Cron Expression Format

Format: `minute hour dayOfMonth month dayOfWeek`

**IMPORTANT**: Cron expressions use the user's **local timezone**, NOT UTC.

### Common Patterns
```
0 9 * * *       → Every day at 9:00 AM
0 9 * * 1-5     → Weekdays at 9:00 AM
30 8 * * 1      → Every Monday at 8:30 AM
0 0 1 * *       → First day of every month at midnight
0 */2 * * *     → Every 2 hours
0 9,17 * * *    → At 9 AM and 5 PM daily
```

## Writing Task Prompts

### Best Practices
1. **Self-contained**: The prompt must include ALL context needed
2. **No session references**: Don't reference conversation history or session state
3. **Specific actions**: Clearly describe what to check, create, or update
4. **Output format**: Specify how results should be presented

### Example — Daily Stand-up
```
taskId: daily-standup
description: Generate a daily standup summary from git commits
cronExpression: 0 9 * * 1-5
prompt: |
  Check the git log for commits made yesterday in the workspace.
  Summarize what was accomplished, what's in progress, and any blockers.
  Format as a brief standup update with 3 sections:
  - Done yesterday
  - Planned today
  - Blockers
```

### Example — Weekly Report
```
taskId: weekly-report
description: Create a weekly progress report
cronExpression: 0 17 * * 5
prompt: |
  Analyze the git commits and file changes from the past week.
  Create a markdown report summarizing:
  - Key accomplishments
  - Files changed (grouped by area)
  - Lines added/removed
  Save as weekly-report-{date}.md in the workspace.
```

## Managing Scheduled Tasks

- **List**: Use `list_scheduled_tasks` to see all tasks
- **Update**: Use `update_scheduled_task` to change schedule or prompt
- **Pause**: Set `enabled: false` to pause automatic runs
- **Delete**: Use `delete_scheduled_task` to remove
- **Run now**: Use `run_scheduled_task` to execute immediately

## Ad-Hoc Tasks

Omit the `cronExpression` to create a task that can only be run manually.
Useful for tasks that need to be triggered on demand.
