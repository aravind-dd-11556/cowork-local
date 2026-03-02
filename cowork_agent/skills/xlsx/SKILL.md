---
name: xlsx
description: "Excel Spreadsheet Handler: Comprehensive Microsoft Excel (.xlsx) document creation, editing, and analysis with support for formulas, formatting, data analysis, and visualization"
---

# Excel Spreadsheet Skill (xlsx)

MANDATORY TRIGGERS: Excel, spreadsheet, .xlsx, data table, budget, financial model, chart, graph, tabular data, xls

## Technology Stack

- **Primary**: Python `openpyxl` for Excel manipulation
- **Analysis**: Python `pandas` for data transformation
- **Charts**: `openpyxl.chart` for embedded charts

## Quick Start

```python
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Data"
ws['A1'] = "Header"
wb.save('output.xlsx')
```

## Critical Rules

### Formulas
- **ALWAYS use Excel formulas**, never pre-computed values
- Example: Use `=SUM(B2:B10)` not `=45`
- Example: Use `=AVERAGE(C2:C100)` not `=23.5`
- Formulas must be valid Excel syntax (not Python expressions)

### Data Types
- Numbers as numbers (not strings)
- Dates as datetime objects
- Currencies formatted with number format (`'$#,##0.00'`)

## Formatting Best Practices

### Headers
```python
header_font = Font(bold=True, size=12, color="FFFFFF")
header_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
for cell in ws[1]:
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = Alignment(horizontal='center')
```

### Column Widths
- Auto-fit based on content length
- Minimum 10 characters wide
- Maximum 50 characters wide

### Number Formats
- Currency: `'$#,##0.00'`
- Percentage: `'0.00%'`
- Date: `'YYYY-MM-DD'`
- Integer: `'#,##0'`

## Charts

```python
from openpyxl.chart import BarChart, Reference

chart = BarChart()
chart.title = "Sales by Month"
data = Reference(ws, min_col=2, min_row=1, max_col=2, max_row=13)
cats = Reference(ws, min_col=1, min_row=2, max_row=13)
chart.add_data(data, titles_from_data=True)
chart.set_categories(cats)
ws.add_chart(chart, "D2")
```

## Reading Existing Files

```python
import pandas as pd
df = pd.read_excel('input.xlsx', sheet_name='Sheet1')
# Process data
df.to_excel('output.xlsx', index=False)
```

## Installation

```bash
pip install openpyxl pandas --break-system-packages
```
