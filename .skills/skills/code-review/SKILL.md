# Code Review Skill

**Trigger:** code review, review code, review changes, pre-commit, audit code

Perform technical code review on recently changed files.

## Core Principles

- Simplicity is the ultimate sophistication — every line should justify its existence
- Code is read far more often than it's written — optimize for readability
- The best code is often the code you don't write
- Elegance emerges from clarity of intent and economy of expression

## What to Review

Start by gathering codebase context to understand the codebase standards and patterns.

Examine:
- `CLAUDE.md` (if present)
- `README.md`
- Key files in the `/core` module
- Documented standards in the `/docs` directory

Then gather changes:

```bash
git status
git diff HEAD
git diff --stat HEAD
git ls-files --others --exclude-standard
```

Read each changed/new file in its entirety (not just the diff) to understand full context.

## Analysis Checklist

For each changed or new file, analyze for:

### 1. Logic Errors
- Off-by-one errors
- Incorrect conditionals
- Missing error handling
- Race conditions

### 2. Security Issues
- SQL injection vulnerabilities
- XSS vulnerabilities
- Insecure data handling
- Exposed secrets or API keys

### 3. Performance Problems
- N+1 queries
- Inefficient algorithms
- Memory leaks
- Unnecessary computations

### 4. Code Quality
- Violations of DRY principle
- Overly complex functions (>50 lines or >5 nesting levels)
- Poor naming
- Missing type hints/annotations

### 5. Adherence to Codebase Standards
- Consistent import ordering
- Docstrings on public classes and methods
- Test coverage for new functionality
- Error handling patterns match existing code

## Verify Issues Are Real

- Run specific tests for issues found
- Confirm type errors are legitimate
- Validate security concerns with context

## Output Format

Save results to `.agents/code-reviews/[date]-[short-desc].md`

**Stats:**
- Files Modified: N
- Files Added: N
- Files Deleted: N
- New lines: N
- Deleted lines: N

**For each issue found:**

```
severity: critical|high|medium|low
file: path/to/file.py
line: 42
issue: [one-line description]
detail: [explanation of why this is a problem]
suggestion: [how to fix it]
```

If no issues found: "Code review passed. No technical issues detected."

## Important

- Be specific (line numbers, not vague complaints)
- Focus on real bugs, not style preferences
- Suggest fixes, don't just complain
- Flag security issues as CRITICAL
- Only flag issues that would break functionality or introduce vulnerabilities
