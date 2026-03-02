---
name: docx
description: "Word Document Handler: Comprehensive Microsoft Word (.docx) document creation, editing, and analysis with support for tracked changes, comments, formatting preservation, and text extraction"
---

# Word Document Skill (docx)

MANDATORY TRIGGERS: Word, document, .docx, report, letter, memo, manuscript, essay, paper, article, writeup, documentation

## Technology Stack

- **Primary**: Python `python-docx` library
- **Fallback**: Node.js `docx` library (for advanced features)
- **XML manipulation**: Direct OOXML editing for features not covered by libraries

## Quick Start

```python
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

doc = Document()
doc.add_heading('Title', level=0)
doc.add_paragraph('Content here.')
doc.save('output.docx')
```

## Creating Documents — Best Practices

### Structure
1. Always set document properties (title, author, subject)
2. Use heading levels consistently (H1 for main sections, H2 for subsections)
3. Add a Table of Contents if the document has 3+ sections
4. Use page breaks between major sections

### Formatting Rules
- **Font**: Use professional fonts (Calibri, Arial, Times New Roman)
- **Size**: Body text 11pt, headings proportionally larger
- **Spacing**: 1.15 or 1.5 line spacing for readability
- **Margins**: Standard 1-inch margins unless specified otherwise

### Tables
- Always include header rows with bold formatting
- Use consistent column widths
- Apply borders and shading for readability

### Headers & Footers
- Include page numbers in footer
- Add document title in header for multi-page docs

## Editing Existing Documents

1. **Read first**: Always read the file to understand structure
2. **Preserve formatting**: When editing, maintain existing styles
3. **Track changes**: Use tracked changes when modifying shared documents
4. **Comments**: Add comments for review when appropriate

## Common Patterns

### Report Template
```python
doc = Document()
# Title page
doc.add_heading('Report Title', 0)
doc.add_paragraph(f'Prepared by: {author}')
doc.add_paragraph(f'Date: {date}')
doc.add_page_break()

# Table of Contents placeholder
doc.add_heading('Table of Contents', 1)
doc.add_paragraph('(Update after document is complete)')
doc.add_page_break()

# Sections
for section_title, content in sections:
    doc.add_heading(section_title, 1)
    doc.add_paragraph(content)
```

### Letter Template
```python
doc = Document()
doc.add_paragraph(sender_address)
doc.add_paragraph(date)
doc.add_paragraph(recipient_address)
doc.add_paragraph(f'Dear {recipient_name},')
doc.add_paragraph(body)
doc.add_paragraph('Sincerely,')
doc.add_paragraph(sender_name)
```

## Installation

```bash
pip install python-docx --break-system-packages
```
