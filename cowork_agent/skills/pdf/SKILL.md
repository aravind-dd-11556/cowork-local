---
name: pdf
description: "PDF Processing: Comprehensive PDF manipulation toolkit for extracting text and tables, creating new PDFs, merging/splitting documents, and handling forms"
---

# PDF Skill

MANDATORY TRIGGERS: PDF, .pdf, form, extract, merge, split

## Technology Stack

- **Reading/Extraction**: `pdfplumber` (text + tables)
- **Merging/Splitting**: `pypdf` (formerly PyPDF2)
- **Creating New PDFs**: `reportlab`
- **OCR**: `pytesseract` + `pdf2image` (for scanned PDFs)

## Quick Start — Reading

```python
import pdfplumber

with pdfplumber.open('input.pdf') as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        tables = page.extract_tables()
```

## Quick Start — Creating

```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

c = canvas.Canvas('output.pdf', pagesize=letter)
c.setFont('Helvetica', 12)
c.drawString(1*inch, 10*inch, 'Hello, World!')
c.save()
```

## Common Operations

### Merge PDFs
```python
from pypdf import PdfReader, PdfWriter

writer = PdfWriter()
for pdf_path in pdf_files:
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        writer.add_page(page)
writer.write('merged.pdf')
```

### Split PDF
```python
from pypdf import PdfReader, PdfWriter

reader = PdfReader('input.pdf')
for i, page in enumerate(reader.pages):
    writer = PdfWriter()
    writer.add_page(page)
    writer.write(f'page_{i+1}.pdf')
```

### Extract Tables
```python
import pdfplumber
import pandas as pd

with pdfplumber.open('input.pdf') as pdf:
    tables = []
    for page in pdf.pages:
        page_tables = page.extract_tables()
        for table in page_tables:
            df = pd.DataFrame(table[1:], columns=table[0])
            tables.append(df)
```

### Fill PDF Forms
```python
from pypdf import PdfReader, PdfWriter

reader = PdfReader('form.pdf')
writer = PdfWriter()
writer.append(reader)

writer.update_page_form_field_values(
    writer.pages[0],
    {'field_name': 'value'}
)
writer.write('filled_form.pdf')
```

## Best Practices

1. **Always check page count** before processing large PDFs
2. **Use pdfplumber** for text extraction (more accurate than pypdf)
3. **Use pypdf** for structural operations (merge, split, rotate)
4. **Use reportlab** for creating new PDFs from scratch
5. **Handle encoding** — some PDFs have unusual character encodings

## Installation

```bash
pip install pdfplumber pypdf reportlab --break-system-packages
```
