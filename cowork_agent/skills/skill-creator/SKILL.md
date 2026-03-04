---
name: skill-creator
description: "Create new skills, modify and improve existing skills, and measure skill performance. Use when users want to create a skill from scratch, edit, or optimize an existing skill, run evals to test a skill, benchmark skill performance with variance analysis, or optimize a skill's description for better triggering accuracy."
---

# Skill Creator — Meta-Skill

## Purpose

Create, test, and optimize skills in the cowork_agent framework.

## Skill File Structure

```
skills/
  my-skill/
    SKILL.md          # Required: instructions and metadata
    examples/         # Optional: example inputs/outputs
    templates/        # Optional: file templates
    scripts/          # Optional: supporting scripts
    agents/           # Optional: agent prompt files
```

## SKILL.md Format

```markdown
---
name: my-skill
description: "Clear description of what this skill does and when to trigger it"
---

# Skill Title

MANDATORY TRIGGERS: keyword1, keyword2, keyword3

## Technology Stack
- List libraries and tools used

## Quick Start
- Minimal working example

## Best Practices
- Rules and guidelines

## Common Patterns
- Reusable code/workflow patterns

## Installation
- Dependencies to install
```

## Creating a New Skill

### Step 1: Define the Skill
- **Name**: Lowercase, hyphenated (e.g., `data-viz`, `email-template`)
- **Description**: One clear sentence — include what triggers it AND what it should NOT trigger on
- **Triggers**: 5-10 keywords that should activate this skill

### Step 2: Write the SKILL.md
1. Start with frontmatter (name, description)
2. Add MANDATORY TRIGGERS line
3. Document the technology stack
4. Provide quick-start code examples
5. List best practices and critical rules
6. Include common patterns with full code
7. Add installation instructions

### Step 3: Test the Skill
- Verify trigger keywords match correctly
- Test with real user messages
- Check that content provides enough guidance for the LLM

### Step 4: Add Supporting Files (Optional)
- Scripts for automation (e.g., validation, conversion)
- Agent prompts for specialized sub-tasks
- Templates for common output formats

## Optimizing a Skill

### Trigger Optimization
1. Analyze false positives (triggers when it shouldn't)
2. Analyze false negatives (doesn't trigger when it should)
3. Adjust keywords to improve precision and recall
4. Test with diverse phrasings of the same intent

### Content Optimization
1. Are instructions clear and actionable?
2. Do code examples work correctly?
3. Are edge cases covered?
4. Are critical rules prominently listed?
5. Is there a QA/verification step?

### Description Optimization
The description field in frontmatter is critical for trigger accuracy:
- Include positive triggers: "Use when..."
- Include negative triggers: "Do NOT use for..."
- Be specific about file types and contexts

## Eval Framework

### Running Evals
```python
test_messages = [
    ("create a word document", "docx"),
    ("make a spreadsheet", "xlsx"),
    ("build a presentation", "pptx"),
    ("what is python?", None),
    ("merge two PDFs", "pdf"),
    ("create a video animation", "remotion"),
]

for message, expected_skill in test_messages:
    matched = skill_registry.match_skills(message)
    actual = matched[0].name if matched else None
    status = "PASS" if actual == expected_skill else "FAIL"
    print(f"[{status}] '{message}' -> expected={expected_skill}, got={actual}")
```

### Variance Analysis
- Run eval suite multiple times
- Measure consistency of matching
- Track improvements across iterations
- Compare precision/recall before and after changes

### Benchmarking
```python
import time

results = []
for message, expected in test_messages:
    start = time.time()
    matched = skill_registry.match_skills(message)
    elapsed = time.time() - start
    results.append({
        'message': message,
        'expected': expected,
        'actual': matched[0].name if matched else None,
        'correct': (matched[0].name if matched else None) == expected,
        'latency_ms': elapsed * 1000,
    })

correct = sum(1 for r in results if r['correct'])
print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.0f}%)")
print(f"Avg latency: {sum(r['latency_ms'] for r in results)/len(results):.1f}ms")
```

## Agent Prompts

For complex skills that need sub-agents, create prompt files in `agents/`:

### agents/grader.md
Evaluates skill output quality on a 1-5 scale.

### agents/comparator.md
Compares two skill versions side-by-side.

### agents/analyzer.md
Analyzes skill performance data and suggests improvements.
