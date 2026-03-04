---
name: pptx
description: "PowerPoint Suite: Microsoft PowerPoint (.pptx) presentation creation, editing, and analysis.\n  MANDATORY TRIGGERS: PowerPoint, presentation, .pptx, slides, slide deck, pitch deck, ppt, slideshow, deck"
---

# PPTX Skill

## Quick Reference

| Task | Guide |
|------|-------|
| Read/analyze content | `python-pptx` — see Reading Content |
| Create from scratch | Use `python-pptx` — see Creating Presentations |
| Extract & verify | Content QA with python-pptx text extraction |
| Convert to images | LibreOffice + pdftoppm — see Converting to Images |

---

## Reading Content

Extract and analyze presentation content using python-pptx:

```python
from pptx import Presentation

prs = Presentation('input.pptx')
for slide_num, slide in enumerate(prs.slides, 1):
    print(f"Slide {slide_num}:")
    for shape in slide.shapes:
        if shape.has_text_frame:
            print(f"  {shape.text_frame.text}")
        if shape.has_table:
            table = shape.table
            for row in table.rows:
                print([cell.text for cell in row.cells])
```

**Extract specific content:**

```python
# Get all text from a slide
def get_slide_text(slide):
    text_parts = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            text_parts.append(shape.text_frame.text)
    return "\n".join(text_parts)

# Get notes
notes_text = slide.notes_slide.notes_text_frame.text

# Get slide dimensions
width_inches = prs.slide_width / 914400  # Convert EMU to inches
height_inches = prs.slide_height / 914400
```

**Inspect shapes and styles:**

```python
for shape in slide.shapes:
    print(f"Shape type: {shape.shape_type}")
    if hasattr(shape, 'fill'):
        print(f"Fill type: {shape.fill.type}")
    if shape.has_text_frame:
        for paragraph in shape.text_frame.paragraphs:
            for run in paragraph.runs:
                print(f"Text: {run.text}, Font: {run.font.name}, Size: {run.font.size}")
```

---

## Design Philosophy

**Don't create boring slides.** Plain bullets on white won't impress anyone. Design matters.

### Foundational Principles

A great presentation is visually cohesive and topically aligned. Your color palette, typography, and layout choices should all reinforce the message. The difference between forgettable and memorable is usually attention to proportion, contrast, and consistency.

### Color Palettes

Pick colors that match your topic — don't default to generic blue. Strong color choices communicate professionalism and confidence.

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

**Color strategy:** Dominance over equality. One color should dominate (60-70%), with 1-2 supporting tones and one sharp accent for callouts. Dark backgrounds work for title and conclusion slides; light backgrounds for content. Use contrast to guide the viewer's eye.

### Layout Strategies

Every slide needs a visual element — image, chart, icon, or shape. Repetition without variety is death by bullet points.

**Effective layouts:** Two-column designs (text left, illustration right) work reliably. Icon + text rows (icon in colored circle, bold header, description) create visual rhythm. 2x2 or 2x3 grids balance information density. Half-bleed images with content overlay add sophistication.

**Data display:** Large stat callouts (60-72pt numbers with small labels below) command attention. Comparison columns (before/after, pros/cons) make trade-offs clear. Timeline or process flows (numbered steps, arrows) show progression.

### Typography

Intentional font pairing elevates the entire deck.

| Header Font | Body Font |
|-------------|-----------|
| Georgia | Calibri |
| Arial Black | Arial |
| Calibri | Calibri Light |
| Cambria | Calibri |
| Trebuchet MS | Calibri |
| Palatino | Garamond |

**Sizing hierarchy:**
- Slide title: 36-44pt bold
- Section header: 20-24pt bold
- Body text: 14-16pt regular
- Captions: 10-12pt muted color

Left-align body text and lists; center only titles. Never compromise readability for aesthetics.

### Spacing & Margins

Breathing room separates good design from cramped design. Maintain 0.5" minimum margins. Use 0.3-0.5" gaps between content blocks. Consistent spacing signals intentionality.

### Critical Avoidances

Don't repeat the same layout — vary columns, cards, and callouts across slides. Never center body text; left-align paragraphs and lists. Don't skimp on size contrast — titles need 36pt+ versus 14-16pt body. Don't default to blue; pick colors reflecting the topic. Never create text-only slides; add images, icons, charts. NEVER use accent lines under titles — hallmark of AI-generated slides. Avoid generic placeholder imagery; thoughtful visuals (or none) beats obvious stock photos.

---

## Creating Presentations

### Basic Setup

Always set slide dimensions **before** adding slides.

```python
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Create presentation
prs = Presentation()
prs.slide_width = Inches(13.333)  # Widescreen 16:9
prs.slide_height = Inches(7.5)

# Add blank slide (avoids placeholder conflicts)
slide = prs.slides.add_slide(prs.slide_layouts[6])

prs.save('output.pptx')
```

**Critical:** Use `prs.slide_layouts[6]` (blank layout) to avoid placeholder text boxes interfering with your positioning.

### Text & Formatting

**Basic title slide:**

```python
slide = prs.slides.add_slide(prs.slide_layouts[6])

# Add title textbox
title_box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(11.333), Inches(2))
text_frame = title_box.text_frame
text_frame.word_wrap = True

# Access first paragraph (already exists, don't add_paragraph)
p = text_frame.paragraphs[0]
p.text = "Presentation Title"
p.font.size = Pt(44)
p.font.bold = True
p.font.color.rgb = RGBColor(0x1E, 0x27, 0x61)
p.alignment = PP_ALIGN.CENTER

# Add subtitle
subtitle_box = slide.shapes.add_textbox(Inches(1), Inches(4.2), Inches(11.333), Inches(1))
subtitle = subtitle_box.text_frame.paragraphs[0]
subtitle.text = "Your compelling subtitle"
subtitle.font.size = Pt(24)
subtitle.font.color.rgb = RGBColor(0xCA, 0xDC, 0xFC)
subtitle.alignment = PP_ALIGN.CENTER

prs.save('output.pptx')
```

**Rich text formatting:**

```python
textbox = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(2))
tf = textbox.text_frame
tf.word_wrap = True

# Multiple runs in same paragraph
p = tf.paragraphs[0]
p.text = "This is "

run1 = p.add_run()
run1.text = "bold"
run1.font.bold = True
run1.font.size = Pt(14)

run2 = p.add_run()
run2.text = " and this is "
run2.font.size = Pt(14)

run3 = p.add_run()
run3.text = "italic"
run3.font.italic = True
run3.font.size = Pt(14)
run3.font.color.rgb = RGBColor(0xF9, 0x61, 0x67)
```

**Multi-line text with margins:**

```python
textbox = slide.shapes.add_textbox(Inches(1.5), Inches(1.5), Inches(10.333), Inches(4))
tf = textbox.text_frame
tf.word_wrap = True
tf.margin_top = Inches(0.15)
tf.margin_bottom = Inches(0.15)
tf.margin_left = Inches(0.2)
tf.margin_right = Inches(0.2)

p = tf.paragraphs[0]
p.text = "Key insight goes here"
p.font.size = Pt(16)
p.font.color.rgb = RGBColor(0x2F, 0x3C, 0x7E)
p.line_spacing = 1.4
```

### Lists & Bullets

**Correct way (using text_frame hierarchy):**

```python
textbox = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(10.333), Inches(4))
tf = textbox.text_frame
tf.word_wrap = True

# First paragraph (no bullet yet)
p = tf.paragraphs[0]
p.text = "Key Points"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = RGBColor(0x1E, 0x27, 0x61)
p.level = 0

# Add bullet items
for item in ["First point here", "Second key insight", "Third important fact"]:
    p = tf.add_paragraph()
    p.text = item
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(0x36, 0x36, 0x36)
    p.level = 1
    p.space_before = Pt(6)
    p.space_after = Pt(6)

# Sub-bullets
p = tf.add_paragraph()
p.text = "Supporting detail"
p.level = 2
p.font.size = Pt(12)
```

**WRONG way (don't do this):**

```python
# BAD: Using Unicode bullets manually
p.text = "• Manual bullet text"  # DON'T DO THIS

# BAD: Concatenating text instead of adding paragraphs
tf.text = "Point 1\n• Point 2"  # DON'T DO THIS

# BAD: Trying to create bullets in a single paragraph
p.text = "Point 1\nPoint 2\nPoint 3"  # Just text, not bullets
```

### Shapes

**Rectangle with fill and styling:**

```python
from pptx.enum.shapes import MSO_SHAPE

# Rectangle shape
shape = slide.shapes.add_shape(
    MSO_SHAPE.RECTANGLE,
    Inches(1), Inches(1),
    Inches(5), Inches(2)
)

# Solid fill (call .solid() BEFORE setting color)
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0x1E, 0x27, 0x61)

# Border styling
shape.line.color.rgb = RGBColor(0xCA, 0xDC, 0xFC)
shape.line.width = Pt(2)

# Add text inside shape
tf = shape.text_frame
tf.word_wrap = True
tf.vertical_anchor = 1  # Middle alignment
p = tf.paragraphs[0]
p.text = "Shaped content"
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
p.alignment = PP_ALIGN.CENTER
```

**Oval/circle with transparency:**

```python
# Oval shape
oval = slide.shapes.add_shape(
    MSO_SHAPE.OVAL,
    Inches(2), Inches(2),
    Inches(2), Inches(2)
)

oval.fill.solid()
oval.fill.fore_color.rgb = RGBColor(0xF9, 0x61, 0x67)
oval.fill.transparency = 0.2  # 20% transparency

# No border
oval.line.fill.background()
```

**Rounded rectangle:**

```python
rounded = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE,
    Inches(1), Inches(1),
    Inches(4), Inches(1.5)
)

rounded.fill.solid()
rounded.fill.fore_color.rgb = RGBColor(0x97, 0xBC, 0x62)
rounded.adjustments[0] = 0.1  # Corner radius

rounded.line.color.rgb = RGBColor(0x2C, 0x5F, 0x2D)
rounded.line.width = Pt(1)
```

**Line connector:**

```python
line = slide.shapes.add_connector(1, Inches(2), Inches(2), Inches(5), Inches(4))
line.line.color.rgb = RGBColor(0xCA, 0xDC, 0xFC)
line.line.width = Pt(2)
```

**Full-slide background shape:**

```python
bg = slide.shapes.add_shape(
    MSO_SHAPE.RECTANGLE,
    Inches(0), Inches(0),
    prs.slide_width, prs.slide_height
)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0x1E, 0x27, 0x61)
bg.line.fill.background()  # No border

# Send to back so content appears on top
slide.shapes._spTree.remove(bg._element)
slide.shapes._spTree.insert(2, bg._element)
```

### Images

**From file:**

```python
img_path = '/path/to/image.jpg'
pic = slide.shapes.add_picture(
    img_path,
    Inches(1), Inches(1),
    width=Inches(6)  # Height scales automatically
)

# Or with explicit height
pic = slide.shapes.add_picture(
    img_path,
    Inches(1), Inches(1),
    height=Inches(4)
)
```

**From URL (download first):**

```python
import requests
from io import BytesIO

url = 'https://example.com/image.jpg'
response = requests.get(url)
img_stream = BytesIO(response.content)

pic = slide.shapes.add_picture(
    img_stream,
    Inches(1), Inches(1),
    width=Inches(6)
)
```

**From base64:**

```python
import base64
from io import BytesIO

base64_string = "iVBORw0KGgoAAAANS..."  # Your base64 image data
img_data = base64.b64decode(base64_string)
img_stream = BytesIO(img_data)

pic = slide.shapes.add_picture(
    img_stream,
    Inches(1), Inches(1),
    width=Inches(5)
)
```

**Aspect ratio calculation:**

```python
from PIL import Image
from io import BytesIO

def add_image_maintain_aspect(slide, img_path, left, top, width_inches):
    """Add image maintaining aspect ratio"""
    img = Image.open(img_path)
    aspect_ratio = img.height / img.width
    height_inches = width_inches * aspect_ratio

    pic = slide.shapes.add_picture(
        img_path, Inches(left), Inches(top),
        width=Inches(width_inches)
    )
    return pic, height_inches

# Usage
pic, height = add_image_maintain_aspect(slide, 'photo.jpg', 1, 1, 5)
```

### Slide Backgrounds

**Solid color background:**

```python
# Method 1: Background shape
bg = slide.shapes.add_shape(
    MSO_SHAPE.RECTANGLE,
    Inches(0), Inches(0),
    prs.slide_width, prs.slide_height
)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0x1E, 0x27, 0x61)
bg.line.fill.background()

# Move to back
slide.shapes._spTree.remove(bg._element)
slide.shapes._spTree.insert(2, bg._element)
```

**Image background:**

```python
# Add full-slide image
bg_pic = slide.shapes.add_picture(
    'background.jpg',
    Inches(0), Inches(0),
    width=prs.slide_width,
    height=prs.slide_height
)

# Move to back
slide.shapes._spTree.remove(bg_pic._element)
slide.shapes._spTree.insert(2, bg_pic._element)

# Add semi-transparent overlay for text readability
overlay = slide.shapes.add_shape(
    MSO_SHAPE.RECTANGLE,
    Inches(0), Inches(0),
    prs.slide_width, prs.slide_height
)
overlay.fill.solid()
overlay.fill.fore_color.rgb = RGBColor(0x00, 0x00, 0x00)
overlay.fill.transparency = 0.4  # 40% transparent
overlay.line.fill.background()

slide.shapes._spTree.remove(overlay._element)
slide.shapes._spTree.insert(3, overlay._element)
```

### Tables

**Basic table:**

```python
rows, cols = 4, 3
left = Inches(1)
top = Inches(1.5)
width = Inches(10.333)
height = Inches(3)

table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
table = table_shape.table

# Set column widths
table.columns[0].width = Inches(3)
table.columns[1].width = Inches(3.666)
table.columns[2].width = Inches(3.667)

# Header row
headers = ["Name", "Role", "Status"]
for i, header_text in enumerate(headers):
    cell = table.cell(0, i)
    cell.text = header_text

    # Format header
    for paragraph in cell.text_frame.paragraphs:
        paragraph.font.bold = True
        paragraph.font.size = Pt(14)
        paragraph.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        paragraph.alignment = PP_ALIGN.CENTER

    # Header background
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(0x1E, 0x27, 0x61)

# Data rows
data = [
    ["Alice Chen", "Product Manager", "Active"],
    ["Bob Smith", "Designer", "Active"],
    ["Carol Davis", "Engineer", "On leave"]
]

for row_idx, row_data in enumerate(data, 1):
    for col_idx, cell_text in enumerate(row_data):
        cell = table.cell(row_idx, col_idx)
        cell.text = cell_text

        # Alternate row colors
        if row_idx % 2 == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xF5, 0xF5, 0xF5)

        # Format text
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = Pt(12)
            if col_idx == 0:
                paragraph.alignment = PP_ALIGN.LEFT
            else:
                paragraph.alignment = PP_ALIGN.CENTER
```

**Advanced table with merged cells:**

```python
# Create 4x4 table
table_shape = slide.shapes.add_table(4, 4, Inches(1), Inches(1), Inches(10.333), Inches(3))
table = table_shape.table

# Merge cells
cell_1_0 = table.cell(1, 0)
cell_1_1 = table.cell(1, 1)
cell_merged = cell_1_0.merge(cell_1_1)
cell_merged.text = "Merged Cell"

# Format merged cell
for p in cell_merged.text_frame.paragraphs:
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

# Vertical merge
cell_1_2 = table.cell(1, 2)
cell_2_2 = table.cell(2, 2)
cell_vmerge = cell_1_2.merge(cell_2_2)
cell_vmerge.text = "Vertical"
```

### Charts

**Bar chart:**

```python
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE

chart_data = CategoryChartData()
chart_data.categories = ['Q1', 'Q2', 'Q3', 'Q4']
chart_data.add_series('Revenue', (1200, 1500, 1800, 2100))
chart_data.add_series('Expenses', (800, 900, 950, 1100))

chart_shape = slide.shapes.add_chart(
    XL_CHART_TYPE.COLUMN_CLUSTERED,
    Inches(1), Inches(1.5),
    Inches(10.333), Inches(4.5),
    chart_data
)

chart = chart_shape.chart
chart.has_legend = True
chart.chart_type = XL_CHART_TYPE.COLUMN_CLUSTERED

# Format chart
plot = chart.plots[0]
plot.vary_by_categories = False

# Series colors
chart.series[0].format.fill.solid()
chart.series[0].format.fill.fore_color.rgb = RGBColor(0x1E, 0x27, 0x61)

chart.series[1].format.fill.solid()
chart.series[1].format.fill.fore_color.rgb = RGBColor(0xF9, 0x61, 0x67)
```

**Line chart:**

```python
chart_data = CategoryChartData()
chart_data.categories = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
chart_data.add_series('Website Traffic', (5200, 6150, 5950, 7100, 7650, 8200))

chart_shape = slide.shapes.add_chart(
    XL_CHART_TYPE.LINE,
    Inches(1), Inches(1.5),
    Inches(10.333), Inches(4.5),
    chart_data
)

chart = chart_shape.chart
chart.has_legend = False

# Format line
chart.series[0].format.line.color.rgb = RGBColor(0x02, 0x80, 0x90)
chart.series[0].format.line.width = Pt(2.5)
```

**Pie chart:**

```python
chart_data = CategoryChartData()
chart_data.categories = ['Product A', 'Product B', 'Product C', 'Product D']
chart_data.add_series('Market Share', (35, 25, 20, 20))

chart_shape = slide.shapes.add_chart(
    XL_CHART_TYPE.PIE,
    Inches(2), Inches(1.5),
    Inches(8.333), Inches(4.5),
    chart_data
)

chart = chart_shape.chart
chart.has_legend = True
```

### Speaker Notes

```python
slide = prs.slides.add_slide(prs.slide_layouts[6])

# Add slide content
title = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(11.333), Inches(1))
tf = title.text_frame
p = tf.paragraphs[0]
p.text = "Market Analysis"
p.font.size = Pt(40)

# Add speaker notes
notes_slide = slide.notes_slide
notes_frame = notes_slide.notes_text_frame
notes_frame.text = """Key talking points:
1. Market grew 23% YoY
2. Competitors entering category
3. We have 18-month advantage

Pause for questions after this slide."""

prs.save('output.pptx')
```

---

## Common Pitfalls

### Color Format

**WRONG:** Using hex strings with # prefix
```python
# DON'T DO THIS
shape.fill.fore_color.rgb = "#1E2761"  # String won't work
```

**CORRECT:** Use RGBColor with three 0x values
```python
from pptx.dml.color import RGBColor
shape.fill.fore_color.rgb = RGBColor(0x1E, 0x27, 0x61)
```

### Shape Imports

**WRONG:** Using string shape names
```python
# DON'T DO THIS
shape = slide.shapes.add_shape("RECTANGLE", ...)  # This won't work
```

**CORRECT:** Import MSO_SHAPE enum
```python
from pptx.enum.shapes import MSO_SHAPE
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, ...)
```

### Layout Selection

**WRONG:** Using default/placeholder layouts
```python
# DON'T DO THIS
slide = prs.slides.add_slide(prs.slide_layouts[0])  # Has placeholders
# Text boxes you add will overlap with placeholder text
```

**CORRECT:** Always use blank layout
```python
slide = prs.slides.add_slide(prs.slide_layouts[6])  # Completely blank
```

### Text Frame Paragraphs

**WRONG:** Adding paragraph for first line
```python
# DON'T DO THIS
tf = textbox.text_frame
p = tf.add_paragraph()  # First paragraph already exists
p.text = "This is wrong"
```

**CORRECT:** Use paragraphs[0] for first text
```python
tf = textbox.text_frame
p = tf.paragraphs[0]  # Already exists, just use it
p.text = "Correct approach"

# Then add more paragraphs if needed
p2 = tf.add_paragraph()
p2.text = "New paragraph"
```

### Slide Dimensions

**WRONG:** Setting dimensions after adding slides
```python
prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[6])
prs.slide_width = Inches(13.333)  # Too late!
```

**CORRECT:** Set dimensions immediately
```python
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])
```

### Unit Conversions

**EMU (English Metric Units):** Internal unit, rarely needed directly
- 914,400 EMU = 1 inch
- 12,700 EMU = 1 point

**Inches:** Most intuitive, use for positioning and sizing
```python
Inches(1.5)  # 1.5 inches
```

**Points (Pt):** Use for font sizes
```python
Pt(14)  # 14 point font
```

**When to use which:**
- **Positioning/sizing shapes:** `Inches()`
- **Font size:** `Pt()`
- **Line width:** `Pt()`
- **Margins:** `Inches()`

Never mix units in the same operation. If you need to compare values, convert to the same unit.

### Fill Operations

**WRONG:** Setting color before calling solid()
```python
shape.fill.fore_color.rgb = RGBColor(0x1E, 0x27, 0x61)  # Too early
shape.fill.solid()  # Resets color
```

**CORRECT:** Call solid() first
```python
shape.fill.solid()  # Must call first
shape.fill.fore_color.rgb = RGBColor(0x1E, 0x27, 0x61)
```

### Border/Line Removal

**WRONG:** Setting line width to zero
```python
shape.line.width = 0  # Doesn't actually remove the border
```

**CORRECT:** Use background fill
```python
shape.line.fill.background()  # This removes the border
```

### Font Properties

**WRONG:** Setting font properties on paragraph
```python
p = text_frame.paragraphs[0]
p.font.bold = True  # This is a shortcut for first run only
p.text = "Line 1\nLine 2"  # Bold only applies to first line
```

**CORRECT:** Set on individual runs for multi-line text
```python
p = text_frame.paragraphs[0]
p.text = "Line 1"
run = p.runs[0]
run.font.bold = True  # Now it's truly applied

# For multi-line, use separate paragraphs
p2 = text_frame.add_paragraph()
p2.text = "Line 2"
p2.runs[0].font.bold = True
```

### Object Reuse

**WRONG:** Reusing shape objects across slides
```python
shape = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1), Inches(1), Inches(2), Inches(2))
slide2.shapes.add_shape(shape)  # Won't work across slides
```

**CORRECT:** Create new shapes for each slide
```python
shape1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1), Inches(1), Inches(2), Inches(2))
shape2 = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1), Inches(1), Inches(2), Inches(2))
```

---

## QA (Required)

**Assume there are problems. Your job is to find them systematically.**

### Content QA

Extract all text and verify accuracy:

```python
from pptx import Presentation

def verify_content(pptx_path):
    """Extract and display all slide content"""
    prs = Presentation(pptx_path)
    issues = []

    for slide_num, slide in enumerate(prs.slides, 1):
        print(f"\n=== Slide {slide_num} ===")

        text_found = False
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    print(f"  {text}")
                    text_found = True

            if shape.has_table:
                table = shape.table
                for row in table.rows:
                    print(f"  [{', '.join(cell.text for cell in row.cells)}]")
                    text_found = True

        if not text_found:
            issues.append(f"Slide {slide_num}: No text content found")

    return issues

issues = verify_content('output.pptx')
if issues:
    print("\n!!! ISSUES FOUND !!!")
    for issue in issues:
        print(f"  {issue}")
```

**Check for:**
- Missing content
- Typos (especially in names, numbers, dates)
- Wrong order of slides
- Duplicated content
- Incomplete sentences or cutoff text
- Empty slides that should have content

### Visual QA Checklist

Create or inspect visual representation using the conversion steps below. Check for:

- Overlapping elements (text bleeding through shapes)
- Text overflow or cut off at edges
- Elements too close (< 0.3" gaps between items)
- Uneven gaps between similar elements (inconsistent spacing)
- Insufficient margin from slide edges (< 0.5")
- Low-contrast text on backgrounds (unreadable combinations)
- Leftover placeholder content from template
- Images stretched or distorted
- Tables with misaligned cells
- Charts with cut-off labels or values

### Verification Loop

1. Generate presentation
2. Extract and review all text content programmatically
3. List specific issues found (if none, look harder for spacing/contrast issues)
4. Fix identified issues
5. **Re-verify only the affected slides** (full reverification if major changes)
6. Repeat until no issues found
7. Convert to images for final visual inspection
8. Review image set for layout and visual issues

```python
# Quick verification template
import sys

prs = Presentation('output.pptx')
error_count = 0

for slide_num, slide in enumerate(prs.slides, 1):
    # Check for text content
    text_found = False
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.strip():
            text_found = True
            break

    if not text_found:
        print(f"ERROR: Slide {slide_num} has no text")
        error_count += 1

if error_count > 0:
    print(f"\nFound {error_count} errors. Fix before deployment.")
    sys.exit(1)
else:
    print("Content QA passed ✓")
```

---

## Converting to Images

Convert PPTX to JPEG images for visual inspection. This requires LibreOffice and pdftoppm.

```bash
# Install dependencies
sudo apt-get install libreoffice poppler-utils

# Convert PPTX to PDF
libreoffice --headless --convert-to pdf output.pptx --outdir ./

# Convert PDF to JPEG images (one per slide)
pdftoppm -jpeg -r 150 output.pdf slide

# Result: slide-1.jpg, slide-2.jpg, slide-3.jpg, etc.
```

**Options:**
- `-r 150`: Resolution in DPI (150 is good for screen viewing, 300+ for print)
- `-jpeg`: Output format (use `-png` for lossless)
- `-singlefile`: Output to single file (for single-page PDF)

```bash
# High-quality output for print-ready verification
pdftoppm -jpeg -r 300 output.pdf slide

# Preview in terminal or external viewer
eog slide-1.jpg  # Eye of GNOME image viewer
display slide-1.jpg  # ImageMagick
```

**Batch conversion in Python:**

```python
import subprocess
import os
from pathlib import Path

def convert_pptx_to_images(pptx_path, output_dir='slides', dpi=150):
    """Convert PPTX to image sequence"""

    # Ensure output directory exists
    Path(output_dir).mkdir(exist_ok=True)

    # Convert to PDF
    pdf_path = os.path.join(output_dir, 'presentation.pdf')
    subprocess.run([
        'libreoffice', '--headless', '--convert-to', 'pdf',
        pptx_path, '--outdir', output_dir
    ], check=True)

    # Convert PDF to images
    image_prefix = os.path.join(output_dir, 'slide')
    subprocess.run([
        'pdftoppm', '-jpeg', f'-r', str(dpi),
        pdf_path, image_prefix
    ], check=True)

    # List created images
    images = sorted(Path(output_dir).glob('slide-*.jpg'))
    print(f"Created {len(images)} slides:")
    for img in images:
        print(f"  {img.name}")

    return images

# Usage
images = convert_pptx_to_images('output.pptx')
```

---

## Installation

```bash
# Install python-pptx and optional dependencies
pip install python-pptx Pillow requests --break-system-packages

# For image conversion (optional but recommended for QA)
sudo apt-get install libreoffice poppler-utils

# Verify installation
python -c "from pptx import Presentation; print('python-pptx OK')"
```

---

## Workflow Summary

1. **Setup:** Create presentation, set dimensions, add blank slides
2. **Design:** Use color palettes and spacing guidelines consistently
3. **Create:** Add text, shapes, images, tables, charts as needed
4. **Verify:** Extract text, check for errors, review layout
5. **Convert:** Generate images to visually inspect
6. **QA:** Review images against checklist, fix issues
7. **Re-verify:** Run content extraction again on fixed version
8. **Done:** Save final presentation

Always assume something is broken until proven otherwise.
