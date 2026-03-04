---
name: docx
description: "Word Document Handler: Comprehensive Microsoft Word (.docx) document creation, editing, and analysis with support for tracked changes, comments, formatting preservation, and text extraction.\n  MANDATORY TRIGGERS: Word, document, .docx, report, letter, memo, manuscript, essay, paper, article, writeup, documentation"
---

# DOCX Creation, Editing, and Analysis

## Overview

A .docx file is a ZIP archive containing XML files. This skill covers creating new documents, editing existing ones, and extracting content.

## Quick Reference

| Task | Approach |
|------|----------|
| Read/analyze content | `python-docx` or unpack for raw XML |
| Create new document | Use `docx-js` (Node.js) — see Creating New Documents |
| Edit existing document | Unpack → edit XML → repack — see Editing Existing Documents |
| Simple Python creation | Use `python-docx` — see Python Quick Start |

## Python Quick Start

For simple document creation, `python-docx` works well:

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

---

## Creating New Documents (Node.js — Recommended)

For professional documents with precise control, use JavaScript `docx` library.

### Setup

```javascript
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        Header, Footer, AlignmentType, PageOrientation, LevelFormat, ExternalHyperlink,
        InternalHyperlink, Bookmark, FootnoteReferenceRun, PositionalTab,
        PositionalTabAlignment, PositionalTabRelativeTo, PositionalTabLeader,
        TabStopType, TabStopPosition, Column, SectionType,
        TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType,
        VerticalAlign, PageNumber, PageBreak } = require('docx');
const fs = require('fs');

const doc = new Document({ sections: [{ children: [/* content */] }] });
Packer.toBuffer(doc).then(buffer => fs.writeFileSync("doc.docx", buffer));
```

### Page Size

```javascript
// CRITICAL: docx-js defaults to A4, not US Letter
// Always set page size explicitly for consistent results
sections: [{
  properties: {
    page: {
      size: {
        width: 12240,   // 8.5 inches in DXA
        height: 15840   // 11 inches in DXA
      },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } // 1 inch margins
    }
  },
  children: [/* content */]
}]
```

**Common page sizes (DXA units, 1440 DXA = 1 inch):**

| Paper | Width | Height | Content Width (1" margins) |
|-------|-------|--------|---------------------------|
| US Letter | 12,240 | 15,840 | 9,360 |
| A4 (default) | 11,906 | 16,838 | 9,026 |

**Landscape orientation:** Pass portrait dimensions and let docx-js swap:
```javascript
size: {
  width: 12240,   // Pass SHORT edge as width
  height: 15840,  // Pass LONG edge as height
  orientation: PageOrientation.LANDSCAPE
},
```

### Styles (Override Built-in Headings)

Use Arial as default font (universally supported). Keep titles black.

```javascript
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 180, after: 180 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    children: [
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Title")] }),
    ]
  }]
});
```

### Lists (NEVER use unicode bullets)

```javascript
// WRONG - never manually insert bullet characters
new Paragraph({ children: [new TextRun("• Item")] })  // BAD

// CORRECT - use numbering config with LevelFormat.BULLET
const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [{
    children: [
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("Bullet item")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun("Numbered item")] }),
    ]
  }]
});
```

### Tables

**CRITICAL: Tables need dual widths** — set both `columnWidths` on the table AND `width` on each cell.

```javascript
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [4680, 4680],
  rows: [
    new TableRow({
      children: [
        new TableCell({
          borders,
          width: { size: 4680, type: WidthType.DXA },
          shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({ children: [new TextRun("Cell")] })]
        })
      ]
    })
  ]
})
```

**Width rules:**
- **Always use `WidthType.DXA`** — never `WidthType.PERCENTAGE` (breaks in Google Docs)
- Table width = sum of `columnWidths`
- Cell `width` must match corresponding `columnWidth`
- Cell `margins` are internal padding — reduce content area, not add to cell width

### Images

```javascript
new Paragraph({
  children: [new ImageRun({
    type: "png",  // Required: png, jpg, jpeg, gif, bmp, svg
    data: fs.readFileSync("image.png"),
    transformation: { width: 200, height: 150 },
    altText: { title: "Title", description: "Desc", name: "Name" }
  })]
})
```

### Page Breaks

```javascript
new Paragraph({ children: [new PageBreak()] })
new Paragraph({ pageBreakBefore: true, children: [new TextRun("New page")] })
```

### Hyperlinks

```javascript
// External
new Paragraph({
  children: [new ExternalHyperlink({
    children: [new TextRun({ text: "Click here", style: "Hyperlink" })],
    link: "https://example.com",
  })]
})

// Internal (bookmark + reference)
new Paragraph({ heading: HeadingLevel.HEADING_1, children: [
  new Bookmark({ id: "ch1", children: [new TextRun("Chapter 1")] }),
]})
new Paragraph({ children: [new InternalHyperlink({
  children: [new TextRun({ text: "See Chapter 1", style: "Hyperlink" })],
  anchor: "ch1",
})]})
```

### Footnotes

```javascript
const doc = new Document({
  footnotes: {
    1: { children: [new Paragraph("Source: Annual Report 2024")] },
  },
  sections: [{
    children: [new Paragraph({
      children: [new TextRun("Revenue grew 15%"), new FootnoteReferenceRun(1)],
    })]
  }]
});
```

### Tab Stops

```javascript
new Paragraph({
  children: [new TextRun("Company Name"), new TextRun("\tJanuary 2025")],
  tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
})
```

### Multi-Column Layouts

```javascript
sections: [{
  properties: {
    column: { count: 2, space: 720, equalWidth: true, separate: true },
  },
  children: [/* content flows across columns */]
}]
```

### Table of Contents

```javascript
new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" })
```

### Headers/Footers

```javascript
sections: [{
  headers: {
    default: new Header({ children: [new Paragraph({ children: [new TextRun("Header")] })] })
  },
  footers: {
    default: new Footer({ children: [new Paragraph({
      children: [new TextRun("Page "), new TextRun({ children: [PageNumber.CURRENT] })]
    })] })
  },
  children: [/* content */]
}]
```

---

## Editing Existing Documents

**Follow all 3 steps in order.**

### Step 1: Unpack

```bash
python -c "
import zipfile
with zipfile.ZipFile('document.docx', 'r') as z:
    z.extractall('unpacked/')
print('Unpacked to unpacked/')
"
```

### Step 2: Edit XML

Edit files in `unpacked/word/`. Main content is in `document.xml`.

**CRITICAL: Use smart quotes for new content:**
```xml
<w:t>Here&#x2019;s a quote: &#x201C;Hello&#x201D;</w:t>
```

| Entity | Character |
|--------|-----------|
| `&#x2018;` | ' (left single) |
| `&#x2019;` | ' (right single / apostrophe) |
| `&#x201C;` | " (left double) |
| `&#x201D;` | " (right double) |

### Step 3: Repack

```bash
python -c "
import zipfile, os
with zipfile.ZipFile('output.docx', 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk('unpacked/'):
        for f in files:
            fp = os.path.join(root, f)
            z.write(fp, os.path.relpath(fp, 'unpacked/'))
print('Repacked to output.docx')
"
```

---

## XML Reference

### Tracked Changes

**Insertion:**
```xml
<w:ins w:id="1" w:author="Claude" w:date="2025-01-01T00:00:00Z">
  <w:r><w:t>inserted text</w:t></w:r>
</w:ins>
```

**Deletion:**
```xml
<w:del w:id="2" w:author="Claude" w:date="2025-01-01T00:00:00Z">
  <w:r><w:delText>deleted text</w:delText></w:r>
</w:del>
```

**Minimal edits — only mark what changes:**
```xml
<w:r><w:t>The term is </w:t></w:r>
<w:del w:id="1" w:author="Claude" w:date="...">
  <w:r><w:delText>30</w:delText></w:r>
</w:del>
<w:ins w:id="2" w:author="Claude" w:date="...">
  <w:r><w:t>60</w:t></w:r>
</w:ins>
<w:r><w:t> days.</w:t></w:r>
```

**Deleting entire paragraphs** — mark paragraph mark as deleted:
```xml
<w:p>
  <w:pPr><w:rPr>
    <w:del w:id="1" w:author="Claude" w:date="2025-01-01T00:00:00Z"/>
  </w:rPr></w:pPr>
  <w:del w:id="2" w:author="Claude" w:date="2025-01-01T00:00:00Z">
    <w:r><w:delText>Entire content being deleted...</w:delText></w:r>
  </w:del>
</w:p>
```

### Comments

```xml
<w:commentRangeStart w:id="0"/>
<w:r><w:t>commented text</w:t></w:r>
<w:commentRangeEnd w:id="0"/>
<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="0"/></w:r>
```

### Schema Compliance

- **Element order in `<w:pPr>`**: `<w:pStyle>`, `<w:numPr>`, `<w:spacing>`, `<w:ind>`, `<w:jc>`, `<w:rPr>` last
- **Whitespace**: Add `xml:space="preserve"` to `<w:t>` with leading/trailing spaces
- **RSIDs**: Must be 8-digit hex (e.g., `00AB1234`)

---

## Critical Rules for docx-js

- **Set page size explicitly** — defaults to A4; use US Letter (12240 x 15840 DXA)
- **Landscape: pass portrait dimensions** — docx-js swaps internally
- **Never use `\n`** — use separate Paragraph elements
- **Never use unicode bullets** — use `LevelFormat.BULLET` with numbering config
- **PageBreak must be in Paragraph** — standalone creates invalid XML
- **ImageRun requires `type`** — always specify png/jpg/etc
- **Always set table `width` with DXA** — never use `WidthType.PERCENTAGE`
- **Tables need dual widths** — `columnWidths` array AND cell `width`, both must match
- **Use `ShadingType.CLEAR`** — never SOLID for table shading
- **Never use tables as dividers** — use border on Paragraph instead
- **TOC requires HeadingLevel only** — no custom styles on heading paragraphs
- **Override built-in styles** — use exact IDs: "Heading1", "Heading2", etc.
- **Include `outlineLevel`** — required for TOC (0 for H1, 1 for H2, etc.)

---

## Common Patterns (Python)

### Report Template
```python
from docx import Document
doc = Document()
doc.core_properties.title = "Report Title"
doc.core_properties.author = "Author Name"
doc.add_heading('Report Title', 0)
doc.add_paragraph(f'Prepared by: {author}')
doc.add_page_break()
for section_title, content in sections:
    doc.add_heading(section_title, 1)
    doc.add_paragraph(content)
doc.save('report.docx')
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

---

## Best Practices

### Structure
1. Always set document properties (title, author, subject)
2. Use heading levels consistently (H1 for main, H2 for sub)
3. Add Table of Contents if 3+ sections
4. Use page breaks between major sections

### Formatting
- **Font**: Professional fonts (Calibri, Arial, Times New Roman)
- **Size**: Body 11pt, headings proportionally larger
- **Spacing**: 1.15 or 1.5 line spacing
- **Margins**: Standard 1-inch unless specified

### Tables
- Always include header rows with bold formatting
- Use consistent column widths
- Apply borders and shading for readability

---

## Installation

```bash
pip install python-docx --break-system-packages
npm install -g docx
```
