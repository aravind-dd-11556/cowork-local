---
name: pptx
description: "PowerPoint Suite: Microsoft PowerPoint (.pptx) presentation creation, editing, and analysis"
---

# PowerPoint Skill (pptx)

MANDATORY TRIGGERS: PowerPoint, presentation, .pptx, slides, slide deck, pitch deck, ppt, slideshow, deck

## Technology Stack

- **Primary**: Python `python-pptx` library
- **Alternative**: Node.js `pptxgenjs` for advanced layouts

## Quick Start

```python
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[0])  # Title slide
slide.shapes.title.text = "Presentation Title"
slide.placeholders[1].text = "Subtitle"
prs.save('output.pptx')
```

## Design Guidelines

### Layout Principles
1. **One idea per slide** — don't overcrowd
2. **Consistent fonts** — max 2 font families
3. **Color palette** — stick to 3-5 colors
4. **Whitespace** — leave breathing room (margins of at least 0.5 inches)
5. **Alignment** — use consistent left or center alignment

### Slide Types
- **Title Slide**: Large title, subtitle, optional image
- **Content Slide**: Heading + bullet points or body text
- **Image Slide**: Full or half-page image with caption
- **Two-Column**: Side-by-side comparison or text + image
- **Chart/Data Slide**: Embedded chart with title
- **Section Divider**: Large text, contrasting background

### Typography
- **Title**: 28-36pt, bold
- **Subtitle**: 18-24pt
- **Body**: 14-18pt
- **Footer/Notes**: 10-12pt

### Color Scheme (Professional Default)
- Primary: #2C3E50 (dark blue)
- Accent: #3498DB (bright blue)
- Text: #2C3E50 (dark) or #FFFFFF (on dark backgrounds)
- Background: #FFFFFF or #F5F6FA

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

### Speaker Notes
- Always include speaker notes for context
- Keep notes concise (2-3 sentences per slide)

## Installation

```bash
pip install python-pptx --break-system-packages
```
