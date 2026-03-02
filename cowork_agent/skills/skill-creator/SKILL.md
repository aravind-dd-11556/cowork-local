---
name: skill-creator
description: "Create new skills, modify and improve existing skills, and measure skill performance. Use when users want to create a skill from scratch, edit, or optimize an existing skill, run evals to test a skill, benchmark skill performance with variance analysis, or optimize a skill's description for better triggering accuracy."
---

# Skill Creator — Meta-Skill

MANDATORY TRIGGERS: create skill, new skill, edit skill, optimize skill, skill performance, eval skill

## Purpose

This skill guides the creation, testing, and optimization of other skills in the cowork_agent framework.

## Skill File Structure

Every skill lives in its own directory under `skills/`:

```
skills/
  my-skill/
    SKILL.md          # Required: instructions and metadata
    examples/         # Optional: example inputs/outputs
    templates/        # Optional: file templates
```

## SKILL.md Format

```markdown
---
name: my-skill
description: "Short description of what this skill does"
---

# Skill Title

MANDATORY TRIGGERS: keyword1, keyword2, keyword3

## Technology Stack
- List libraries and tools used

## Quick Start
- Minimal example to get started

## Best Practices
- Rules and guidelines

## Common Patterns
- Reusable code/workflow patterns

## Installation
- Any dependencies to install
```

## Creating a New Skill

### Step 1: Define the Skill
- **Name**: Lowercase, hyphenated (e.g., `data-viz`, `email-template`)
- **Description**: One clear sentence describing what it does
- **Triggers**: 5-10 keywords that should activate this skill

### Step 2: Write the SKILL.md
- Start with frontmatter (name, description)
- Add MANDATORY TRIGGERS line
- Document the technology stack
- Provide quick-start code examples
- List best practices and common patterns
- Include installation instructions

### Step 3: Test the Skill
- Verify trigger keywords match correctly
- Test with real user messages
- Check that the skill content provides enough guidance

## Optimizing a Skill

### Trigger Optimization
1. Analyze false positives (skill triggers when it shouldn't)
2. Analyze false negatives (skill doesn't trigger when it should)
3. Adjust keywords to improve precision and recall

### Content Optimization
1. Are the instructions clear and actionable?
2. Do the code examples work correctly?
3. Is the technology stack up to date?
4. Are edge cases covered?

## Eval Framework

### Running Evals
```python
# Test skill matching accuracy
test_messages = [
    ("create a word document", "docx"),         # should match docx
    ("make a spreadsheet", "xlsx"),              # should match xlsx
    ("what is python?", None),                    # should match nothing
]

for message, expected_skill in test_messages:
    matched = skill_registry.match_skills(message)
    actual = matched[0].name if matched else None
    assert actual == expected_skill, f"Expected {expected_skill}, got {actual}"
```

### Variance Analysis
- Run the same eval multiple times
- Measure consistency of skill matching
- Track improvements over iterations
