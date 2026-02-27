"""
NotebookEdit Tool — Edit Jupyter notebook (.ipynb) cells.

Mirrors real Cowork's NotebookEdit:
  - Replace cell contents by cell_number (0-indexed) or cell_id
  - Insert new cells at a position
  - Delete cells
  - Supports code and markdown cell types
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

from .base import BaseTool

logger = logging.getLogger(__name__)


class NotebookEditTool(BaseTool):
    name = "notebook_edit"
    description = (
        "Edit Jupyter notebook (.ipynb) cells. Supports replacing cell contents, "
        "inserting new cells, and deleting cells. Cell numbers are 0-indexed. "
        "Use edit_mode='insert' to add a new cell, 'delete' to remove one, "
        "or 'replace' (default) to update existing cell source."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "notebook_path": {
                "type": "string",
                "description": "Absolute path to the .ipynb file",
            },
            "cell_number": {
                "type": "integer",
                "description": "0-indexed cell number to edit. For insert, the new cell is placed at this index.",
            },
            "new_source": {
                "type": "string",
                "description": "The new source content for the cell",
            },
            "cell_type": {
                "type": "string",
                "enum": ["code", "markdown"],
                "description": "Cell type. Required for insert. For replace, defaults to existing cell type.",
            },
            "edit_mode": {
                "type": "string",
                "enum": ["replace", "insert", "delete"],
                "description": "Edit mode: replace (default), insert, or delete",
            },
            "cell_id": {
                "type": "string",
                "description": "Optional cell ID. When inserting, new cell is placed after this cell.",
            },
        },
        "required": ["notebook_path", "new_source"],
    }

    async def execute(
        self,
        notebook_path: str,
        new_source: str = "",
        cell_number: int = 0,
        cell_type: str = "",
        edit_mode: str = "replace",
        cell_id: str = "",
        tool_id: str = "",
        **kwargs,
    ) -> "ToolResult":
        path = Path(notebook_path)

        # Validate path
        if not path.is_absolute():
            return self._error(f"notebook_path must be absolute, got: {notebook_path}", tool_id)

        if not path.exists():
            return self._error(f"Notebook not found: {notebook_path}", tool_id)

        if path.suffix != ".ipynb":
            return self._error(f"Not a Jupyter notebook: {notebook_path}", tool_id)

        # SEC-MEDIUM-2: Check file size before loading (prevent memory exhaustion)
        MAX_NOTEBOOK_SIZE = 10 * 1024 * 1024  # 10 MB
        try:
            file_size = path.stat().st_size
            if file_size > MAX_NOTEBOOK_SIZE:
                return self._error(
                    f"Notebook too large ({file_size / 1024 / 1024:.1f} MB). "
                    f"Maximum allowed: {MAX_NOTEBOOK_SIZE / 1024 / 1024:.0f} MB",
                    tool_id,
                )
        except OSError as e:
            return self._error(f"Cannot stat notebook: {e}", tool_id)

        # Load notebook
        try:
            with open(path, "r", encoding="utf-8") as f:
                notebook = json.load(f)
        except json.JSONDecodeError as e:
            return self._error(f"Invalid notebook JSON: {e}", tool_id)
        except Exception as e:
            return self._error(f"Error reading notebook: {e}", tool_id)

        cells = notebook.get("cells", [])

        # Resolve cell index from cell_id if provided
        if cell_id and edit_mode != "insert":
            found = False
            for i, cell in enumerate(cells):
                if cell.get("id") == cell_id:
                    cell_number = i
                    found = True
                    break
            if not found:
                return self._error(f"Cell with id '{cell_id}' not found", tool_id)

        # Handle edit modes
        if edit_mode == "delete":
            return self._delete_cell(notebook, cells, cell_number, path, tool_id)
        elif edit_mode == "insert":
            return self._insert_cell(notebook, cells, cell_number, cell_id, new_source, cell_type, path, tool_id)
        else:  # replace
            return self._replace_cell(notebook, cells, cell_number, new_source, cell_type, path, tool_id)

    def _replace_cell(self, notebook, cells, cell_number, new_source, cell_type, path, tool_id):
        """Replace an existing cell's source."""
        if cell_number < 0 or cell_number >= len(cells):
            return self._error(
                f"Cell number {cell_number} out of range (0-{len(cells)-1})", tool_id
            )

        cell = cells[cell_number]

        # Update cell type if specified
        if cell_type:
            cell["cell_type"] = cell_type

        # Update source (must be a list of lines for ipynb format)
        cell["source"] = self._to_source_lines(new_source)

        # Clear outputs for code cells
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

        return self._save_notebook(notebook, path, tool_id,
                                   f"Replaced cell {cell_number} ({cell.get('cell_type', 'unknown')})")

    def _insert_cell(self, notebook, cells, cell_number, cell_id, new_source, cell_type, path, tool_id):
        """Insert a new cell."""
        if not cell_type:
            return self._error("cell_type is required for insert mode", tool_id)

        # Determine insert position
        insert_at = cell_number
        if cell_id:
            # Insert after the cell with matching id
            for i, cell in enumerate(cells):
                if cell.get("id") == cell_id:
                    insert_at = i + 1
                    break

        insert_at = max(0, min(insert_at, len(cells)))

        # Create new cell
        new_cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": self._to_source_lines(new_source),
        }

        if cell_type == "code":
            new_cell["outputs"] = []
            new_cell["execution_count"] = None

        cells.insert(insert_at, new_cell)

        return self._save_notebook(notebook, path, tool_id,
                                   f"Inserted {cell_type} cell at position {insert_at}")

    def _delete_cell(self, notebook, cells, cell_number, path, tool_id):
        """Delete a cell."""
        if cell_number < 0 or cell_number >= len(cells):
            return self._error(
                f"Cell number {cell_number} out of range (0-{len(cells)-1})", tool_id
            )

        deleted_type = cells[cell_number].get("cell_type", "unknown")
        cells.pop(cell_number)

        return self._save_notebook(notebook, path, tool_id,
                                   f"Deleted {deleted_type} cell at position {cell_number}")

    def _save_notebook(self, notebook, path, tool_id, message):
        """Save the modified notebook back to disk."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
                f.write("\n")
            return self._success(message, tool_id)
        except Exception as e:
            return self._error(f"Error saving notebook: {e}", tool_id)

    @staticmethod
    def _to_source_lines(source: str) -> list[str]:
        """Convert a string to ipynb source format (list of lines with newlines)."""
        if not source:
            return []
        lines = source.split("\n")
        # Add newline to all lines except the last
        result = [line + "\n" for line in lines[:-1]]
        if lines[-1]:  # Don't add empty string for trailing newline
            result.append(lines[-1])
        return result
