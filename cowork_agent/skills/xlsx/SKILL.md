---
name: xlsx
description: "Excel Spreadsheet Handler: Comprehensive Microsoft Excel (.xlsx) document creation, editing, and analysis with support for formulas, formatting, data analysis, and visualization.\n  MANDATORY TRIGGERS: Excel, spreadsheet, .xlsx, data table, budget, financial model, chart, graph, tabular data, xls"
---

# Requirements for Outputs

## All Excel Files

### Professional Font
- Use consistent, professional font (Arial, Calibri) for all deliverables

### Zero Formula Errors
- Every Excel model MUST be delivered with ZERO formula errors (#REF!, #DIV/0!, #VALUE!, #N/A, #NAME?)

### Preserve Existing Templates
- Study and EXACTLY match existing format when modifying files
- Existing template conventions ALWAYS override these guidelines

## Financial Models

### Color Coding Standards (Industry-Standard)

- **Blue text (RGB: 0,0,255)**: Hardcoded inputs and scenario numbers
- **Black text (RGB: 0,0,0)**: ALL formulas and calculations
- **Green text (RGB: 0,128,0)**: Links from other worksheets in same workbook
- **Red text (RGB: 255,0,0)**: External links to other files
- **Yellow background (RGB: 255,255,0)**: Key assumptions needing attention

### Number Formatting Standards

- **Years**: Format as text strings ("2024" not "2,024")
- **Currency**: Use $#,##0 format; ALWAYS specify units in headers ("Revenue ($mm)")
- **Zeros**: Format all zeros as "-" including percentages ("$#,##0;($#,##0);-")
- **Percentages**: Default to 0.0% (one decimal)
- **Multiples**: Format as 0.0x for valuation multiples (EV/EBITDA, P/E)
- **Negative numbers**: Use parentheses (123) not minus -123

### Formula Construction Rules

- Place ALL assumptions in separate cells, never hardcode in formulas
- Use cell references: `=B5*(1+$B$6)` not `=B5*1.05`
- Document sources for hardcodes: "Source: Company 10-K, FY2024, Page 45"

---

# XLSX Creation, Editing, and Analysis

## CRITICAL: Use Formulas, Not Hardcoded Values

**ALWAYS use Excel formulas instead of calculating values in Python and hardcoding them.**

### WRONG - Hardcoding
```python
total = df['Sales'].sum()
sheet['B10'] = total  # Bad: hardcodes 5000
```

### CORRECT - Using Formulas
```python
sheet['B10'] = '=SUM(B2:B9)'  # Good: Excel calculates
sheet['C5'] = '=(C4-C2)/C2'   # Good: dynamic growth rate
sheet['D20'] = '=AVERAGE(D2:D19)'  # Good: Excel function
```

## Reading and Analyzing Data

### With pandas
```python
import pandas as pd

df = pd.read_excel('file.xlsx')
all_sheets = pd.read_excel('file.xlsx', sheet_name=None)

df.head()
df.info()
df.describe()

df.to_excel('output.xlsx', index=False)
```

### With openpyxl (preserves formulas)
```python
from openpyxl import load_workbook

wb = load_workbook('file.xlsx')
ws = wb.active
print(ws['A1'].value)

# Read calculated values (WARNING: saves will lose formulas)
wb_data = load_workbook('file.xlsx', data_only=True)
```

## Creating New Excel Files

```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, numbers
from openpyxl.utils import get_column_letter

wb = Workbook()
ws = wb.active
ws.title = "Data"

# Headers with formatting
header_font = Font(bold=True, size=12, color="FFFFFF")
header_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
headers = ["Name", "Revenue", "Growth", "Status"]

for col, header in enumerate(headers, 1):
    cell = ws.cell(row=1, column=col, value=header)
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = Alignment(horizontal='center')

# Column widths
for col in range(1, len(headers) + 1):
    ws.column_dimensions[get_column_letter(col)].width = 15

# Number formats
ws['B2'].number_format = '$#,##0.00'
ws['C2'].number_format = '0.0%'

# Formulas
ws['B10'] = '=SUM(B2:B9)'
ws['C10'] = '=AVERAGE(C2:C9)'

wb.save('output.xlsx')
```

## Editing Existing Files

```python
from openpyxl import load_workbook

wb = load_workbook('existing.xlsx')
ws = wb.active

# Modify cells
ws['A1'] = 'New Value'
ws.insert_rows(2)
ws.delete_cols(3)

# Add new sheet
new_ws = wb.create_sheet('Summary')
new_ws['A1'] = 'Summary Data'

wb.save('modified.xlsx')
```

## Charts

```python
from openpyxl.chart import BarChart, LineChart, PieChart, Reference

# Bar chart
chart = BarChart()
chart.title = "Sales by Month"
chart.style = 10
data = Reference(ws, min_col=2, min_row=1, max_col=2, max_row=13)
cats = Reference(ws, min_col=1, min_row=2, max_row=13)
chart.add_data(data, titles_from_data=True)
chart.set_categories(cats)
chart.width = 20
chart.height = 12
ws.add_chart(chart, "D2")

# Line chart
line = LineChart()
line.title = "Trend Analysis"
line.y_axis.title = "Revenue ($)"
line.x_axis.title = "Month"
data = Reference(ws, min_col=2, min_row=1, max_col=3, max_row=13)
line.add_data(data, titles_from_data=True)
ws.add_chart(line, "D18")
```

## Conditional Formatting

```python
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule

# Highlight cells > 1000
ws.conditional_formatting.add('B2:B100',
    CellIsRule(operator='greaterThan', formula=['1000'],
              fill=PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')))

# Color scale (red to green)
ws.conditional_formatting.add('C2:C100',
    ColorScaleRule(start_type='min', start_color='F8696B',
                   mid_type='percentile', mid_value=50, mid_color='FFEB84',
                   end_type='max', end_color='63BE7B'))
```

## Formula Verification Checklist

### Essential Verification
- [ ] **Test 2-3 sample references**: Verify correct values before building full model
- [ ] **Column mapping**: Confirm Excel columns match (column 64 = BL, not BK)
- [ ] **Row offset**: Excel rows are 1-indexed (DataFrame row 5 = Excel row 6)

### Common Pitfalls
- [ ] **NaN handling**: Check with `pd.notna()` before writing
- [ ] **Division by zero**: Check denominators in formulas (#DIV/0!)
- [ ] **Wrong references**: Verify cell references point to intended cells (#REF!)
- [ ] **Cross-sheet references**: Use format `Sheet1!A1`

## Best Practices

### Library Selection
- **pandas**: Data analysis, bulk operations, simple exports
- **openpyxl**: Complex formatting, formulas, Excel-specific features

### Working with openpyxl
- Cell indices are 1-based (row=1, column=1 = A1)
- `data_only=True` reads values but **saving will lose formulas**
- Large files: `read_only=True` for reading, `write_only=True` for writing

### Code Style
- Write minimal, concise Python without unnecessary comments
- Add comments to cells with complex formulas
- Document data sources for hardcoded values

---

## Installation

```bash
pip install openpyxl pandas --break-system-packages
```
