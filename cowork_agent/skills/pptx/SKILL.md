---
name: pptx
description: "PowerPoint Suite: Microsoft PowerPoint (.pptx) presentation creation, editing, and analysis.\n  MANDATORY TRIGGERS: PowerPoint, presentation, .pptx, slides, slide deck, pitch deck, ppt, slideshow, deck"
---

# PPTX Skill

## Quick Reference

| Task | Guide |
|------|-------|
| Read/analyze content | `python-pptx` or extract text via unpack |
| Create from scratch | Use `python-pptx` — see Creating Presentations |
| Edit existing | Load with `python-pptx` — see Editing |

## Reading Content

```python
from pptx import Presentation
prs = Presentation('input.pptx')
for slide in prs.slides:
    for shape in slide.shapes:
        if shape.has_text_frame:
            print(shape.text_frame.text)
```

---

## Design Philosophy

**Don't create boring slides.** Plain bullets on white won't impress anyone.

### Before Starting

- **Pick a bold, content-informed color palette**: It should feel designed for THIS topic
- **Dominance over equality**: One color dominates (60-70%), with 1-2 supporting tones and one sharp accent
- **Dark/light contrast**: Dark backgrounds for title + conclusion, light for content
- **Commit to a visual motif**: Pick ONE distinctive element and repeat it

### Color Palettes

Choose colors that match your topic — don't default to generic blue:

| Theme | Primary | Secondary | Accent |
|-------|---------|-----------|--------|
| **Midnight Executive** | `1E2761` (navy) | `CADCFC` (ice blue) | `FFFFFF` (white) |
| **Forest & Moss** | `2C5F2D` (forest) | `97BC62` (moss) | `F5F5F5` (cream) |
| **Coral Energy** | `F96167` (coral) | `F9E795` (gold) | `2F3C7E` (navy) |
| **Warm Terracotta** | `B85042` (terracotta) | `E7E8D1` (sand) | `A7BEAE` (sage) |
| **Ocean Gradient** | `065A82` (deep blue) | `1C7293` (teal) | `21295C` (midnight) |
| **Charcoal Minimal** | `36454F` (charcoal) | `F2F2F2` (off-white) | `212121` (black) |
| **Teal Trust** | `028090` (teal) | `00A896` (seafoam) | `02C39A` (mint) |
| **Berry & Cream** | `6D2E46` (berry) | `A26769` (dusty rose) | `ECE2D0` (cream) |
| **Cherry Bold** | `990011` (cherry) | `FCF6F5` (off-white) | `2F3C7E` (navy) |

### For Each Slide

**Every slide needs a visual element** — image, chart, icon, or shape.

**Layout options:**
- Two-column (text left, illustration right)
- Icon + text rows (icon in colored circle, bold header, description below)
- 2x2 or 2x3 grid (image on one side, content blocks on other)
- Half-bleed image with content overlay

**Data display:**
- Large stat callouts (big numbers 60-72pt with small labels below)
- Comparison columns (before/after, pros/cons)
- Timeline or process flow (numbered steps, arrows)

### Typography

**Choose an interesting font pairing:**

| Header Font | Body Font |
|-------------|-----------|
| Georgia | Calibri |
| Arial Black | Arial |
| Calibri | Calibri Light |
| Cambria | Calibri |
| Trebuchet MS | Calibri |
| Palatino | Garamond |

| Element | Size |
|---------|------|
| Slide title | 36-44pt bold |
| Section header | 20-24pt bold |
| Body text | 14-16pt |
| Captions | 10-12pt muted |

### Spacing

- 0.5" minimum margins
- 0.3-0.5" between content blocks
- Leave breathing room

### Avoid (Common Mistakes)

- **Don't repeat the same layout** — vary columns, cards, and callouts
- **Don't center body text** — left-align paragraphs and lists; center only titles
- **Don't skimp on size contrast** — titles need 36pt+ vs 14-16pt body
- **Don't default to blue** — pick colors reflecting the topic
- **Don't create text-only slides** — add images, icons, charts
- **NEVER use accent lines under titles** — hallmark of AI-generated slides

---

## Creating Presentations

```python
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

prs = Presentation()
prs.slide_width = Inches(13.333)  # Widescreen 16:9
prs.slide_height = Inches(7.5)

# Title Slide
slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
title = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(11), Inches(2))
tf = title.text_frame
p = tf.paragraphs[0]
p.text = "Presentation Title"
p.font.size = Pt(44)
p.font.bold = True
p.font.color.rgb = RGBColor(0x1E, 0x27, 0x61)
p.alignment = PP_ALIGN.CENTER

prs.save('output.pptx')
```

### Adding Shapes with Background Color

```python
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# Full-slide background shape
bg = slide.shapes.add_shape(
    MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
    prs.slide_width, prs.slide_height
)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0x1E, 0x27, 0x61)
bg.line.fill.background()  # No border
```

### Tables

```python
rows, cols = 4, 3
table_shape = slide.shapes.add_table(rows, cols, Inches(1), Inches(2), Inches(8), Inches(3))
table = table_shape.table

# Header row
for i, header in enumerate(["Name", "Role", "Status"]):
    cell = table.cell(0, i)
    cell.text = header
    for paragraph in cell.text_frame.paragraphs:
        paragraph.font.bold = True
        paragraph.font.size = Pt(14)
```

### Charts

```python
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE

chart_data = CategoryChartData()
chart_data.categories = ['Q1', 'Q2', 'Q3', 'Q4']
chart_data.add_series('Revenue', (1200, 1500, 1800, 2100))

chart = slide.shapes.add_chart(
    XL_CHART_TYPE.COLUMN_CLUSTERED,
    Inches(1), Inches(2), Inches(8), Inches(4),
    chart_data
).chart
chart.has_legend = False
```

### Speaker Notes

```python
notes_slide = slide.notes_slide
notes_slide.notes_text_frame.text = "Key talking points for this slide..."
```

---

## Presentation Structure

### Standard Deck (10-15 slides)
1. Title slide
2. Agenda/Overview
3. Key Message 1 (with supporting data)
4. Key Message 2
5. Key Message 3
6. Data/Chart slide
7. Summary/Conclusion
8. Next Steps / Call to Action
9. Q&A / Contact

---

## QA (Required)

**Assume there are problems. Your job is to find them.**

### Content QA

```python
prs = Presentation('output.pptx')
for i, slide in enumerate(prs.slides):
    print(f"--- Slide {i+1} ---")
    for shape in slide.shapes:
        if shape.has_text_frame:
            print(shape.text_frame.text)
```

Check for missing content, typos, wrong order.

### Visual QA Checklist

- Overlapping elements (text through shapes)
- Text overflow or cut off at edges
- Elements too close (< 0.3" gaps)
- Uneven gaps between similar elements
- Insufficient margin from slide edges (< 0.5")
- Low-contrast text on backgrounds
- Leftover placeholder content

### Verification Loop

1. Generate slides
2. Extract and review text content
3. **List issues found** (if none, look harder)
4. Fix issues
5. **Re-verify affected slides**
6. Repeat until clean

---

## Installation

```bash
pip install python-pptx --break-system-packages
```
