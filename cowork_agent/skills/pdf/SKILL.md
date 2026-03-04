---
name: pdf
description: "PDF Processing: Comprehensive PDF manipulation toolkit for extracting text and tables, creating new PDFs, merging/splitting documents, handling forms, OCR, watermarks, encryption.\n  MANDATORY TRIGGERS: PDF, .pdf, form, extract, merge, split"
---

# PDF Processing Guide

## Overview

This guide covers PDF processing using Python libraries and command-line tools.

## Quick Start

```python
from pypdf import PdfReader, PdfWriter

reader = PdfReader("document.pdf")
print(f"Pages: {len(reader.pages)}")

text = ""
for page in reader.pages:
    text += page.extract_text()
```

## Python Libraries

### pypdf — Structural Operations

#### Merge PDFs
```python
from pypdf import PdfWriter, PdfReader

writer = PdfWriter()
for pdf_file in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]:
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        writer.add_page(page)
with open("merged.pdf", "wb") as output:
    writer.write(output)
```

#### Split PDF
```python
reader = PdfReader("input.pdf")
for i, page in enumerate(reader.pages):
    writer = PdfWriter()
    writer.add_page(page)
    with open(f"page_{i+1}.pdf", "wb") as output:
        writer.write(output)
```

#### Extract Metadata
```python
reader = PdfReader("document.pdf")
meta = reader.metadata
print(f"Title: {meta.title}, Author: {meta.author}")
```

#### Rotate Pages
```python
reader = PdfReader("input.pdf")
writer = PdfWriter()
page = reader.pages[0]
page.rotate(90)
writer.add_page(page)
with open("rotated.pdf", "wb") as output:
    writer.write(output)
```

### pdfplumber — Text and Table Extraction

#### Extract Text
```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        print(text)
```

#### Extract Tables
```python
import pandas as pd

with pdfplumber.open("document.pdf") as pdf:
    all_tables = []
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            if table:
                df = pd.DataFrame(table[1:], columns=table[0])
                all_tables.append(df)
    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        combined.to_excel("extracted_tables.xlsx", index=False)
```

### reportlab — Create PDFs

#### Basic PDF
```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

c = canvas.Canvas("hello.pdf", pagesize=letter)
width, height = letter
c.drawString(100, height - 100, "Hello World!")
c.save()
```

#### Multi-Page with Platypus
```python
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("report.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("Report Title", styles['Title']))
story.append(Spacer(1, 12))
story.append(Paragraph("Content here. " * 20, styles['Normal']))
story.append(PageBreak())
story.append(Paragraph("Page 2", styles['Heading1']))

doc.build(story)
```

#### Subscripts and Superscripts

**IMPORTANT**: Never use Unicode subscript/superscript characters in ReportLab. They render as black boxes.

```python
# Use XML markup tags instead
chemical = Paragraph("H<sub>2</sub>O", styles['Normal'])
squared = Paragraph("x<super>2</super>", styles['Normal'])
```

## PDF Form Filling

```python
from pypdf import PdfReader, PdfWriter

reader = PdfReader('form.pdf')
writer = PdfWriter()
writer.append(reader)

# List form fields
for page in reader.pages:
    if '/Annots' in page:
        for annot in page['/Annots']:
            obj = annot.get_object()
            if '/T' in obj:
                print(f"Field: {obj['/T']}, Type: {obj.get('/FT', 'unknown')}")

# Fill fields
writer.update_page_form_field_values(
    writer.pages[0],
    {'field_name': 'value', 'another_field': 'another_value'}
)
writer.write('filled_form.pdf')
```

## Command-Line Tools

### pdftotext (poppler-utils)
```bash
pdftotext input.pdf output.txt
pdftotext -layout input.pdf output.txt      # preserve layout
pdftotext -f 1 -l 5 input.pdf output.txt    # pages 1-5
```

### qpdf
```bash
qpdf --empty --pages file1.pdf file2.pdf -- merged.pdf
qpdf input.pdf --pages . 1-5 -- pages1-5.pdf
qpdf input.pdf output.pdf --rotate=+90:1
qpdf --password=pass --decrypt encrypted.pdf decrypted.pdf
```

## Common Tasks

### OCR Scanned PDFs
```python
import pytesseract
from pdf2image import convert_from_path

images = convert_from_path('scanned.pdf')
text = ""
for i, image in enumerate(images):
    text += f"Page {i+1}:\n"
    text += pytesseract.image_to_string(image)
```

### Add Watermark
```python
watermark = PdfReader("watermark.pdf").pages[0]
reader = PdfReader("document.pdf")
writer = PdfWriter()
for page in reader.pages:
    page.merge_page(watermark)
    writer.add_page(page)
with open("watermarked.pdf", "wb") as output:
    writer.write(output)
```

### Password Protection
```python
writer = PdfWriter()
for page in PdfReader("input.pdf").pages:
    writer.add_page(page)
writer.encrypt("userpassword", "ownerpassword")
with open("encrypted.pdf", "wb") as output:
    writer.write(output)
```

### Extract Images
```bash
pdfimages -j input.pdf output_prefix
```

## Quick Reference

| Task | Best Tool | Command/Code |
|------|-----------|--------------|
| Merge PDFs | pypdf | `writer.add_page(page)` |
| Split PDFs | pypdf | One page per file |
| Extract text | pdfplumber | `page.extract_text()` |
| Extract tables | pdfplumber | `page.extract_tables()` |
| Create PDFs | reportlab | Canvas or Platypus |
| Fill forms | pypdf | `update_page_form_field_values()` |
| OCR scanned | pytesseract | Convert to image first |
| CLI merge | qpdf | `qpdf --empty --pages ...` |

## Installation

```bash
pip install pdfplumber pypdf reportlab --break-system-packages
# For OCR: pip install pytesseract pdf2image --break-system-packages
```
