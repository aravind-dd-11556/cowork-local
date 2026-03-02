"""
File Management Tools — Sprint 32.

Mirrors real Cowork's file management layer:
  - request_cowork_directory: Request access to a directory on the user's computer
  - allow_cowork_file_delete: Request permission to delete files
  - present_files: Display file cards with actions in chat UI

These tools handle the secure file access model where the agent must
request permission before accessing user files or performing destructive
operations.
"""

from __future__ import annotations
import logging
import os
from typing import List, Optional

from .base import BaseTool

logger = logging.getLogger(__name__)


class RequestCoworkDirectoryTool(BaseTool):
    """
    Request access to a directory on the user's computer.

    Shows a directory picker dialog to the user. If they select a directory,
    it will be mounted and made available to the agent. Use this whenever
    the user asks to work with files on their computer and you don't
    currently have access to the relevant location.
    """
    name = "request_cowork_directory"
    description = (
        "Request access to a directory on the user's computer. This will "
        "show a directory picker dialog to the user, and if they select a "
        "directory, it will be mounted and made available to you. Use this "
        "whenever the user asks to work with files on their computer and you "
        "don't currently have access to the relevant location."
    )
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, workspace_dir: str = "", on_directory_requested=None):
        """
        Args:
            workspace_dir: Current workspace directory (if any).
            on_directory_requested: Optional callback invoked when directory
                access is requested. Signature: () -> Optional[str].
                Returns selected path or None if cancelled.
        """
        self._workspace_dir = workspace_dir
        self._on_directory_requested = on_directory_requested
        self._granted_directories: List[str] = []
        if workspace_dir:
            self._granted_directories.append(workspace_dir)

    @property
    def workspace_dir(self) -> str:
        return self._workspace_dir

    @workspace_dir.setter
    def workspace_dir(self, value: str):
        self._workspace_dir = value
        if value and value not in self._granted_directories:
            self._granted_directories.append(value)

    @property
    def granted_directories(self) -> List[str]:
        return list(self._granted_directories)

    async def execute(self, *, progress_callback=None, **kwargs) -> "ToolResult":
        if self._on_directory_requested:
            try:
                selected = self._on_directory_requested()
                if selected:
                    self._workspace_dir = selected
                    if selected not in self._granted_directories:
                        self._granted_directories.append(selected)
                    return self._success(
                        f"Directory access granted: {selected}\n"
                        f"You can now read and write files in this directory.",
                        directory=selected,
                    )
                else:
                    return self._success(
                        "Directory selection was cancelled by the user.",
                        cancelled=True,
                    )
            except Exception as e:
                return self._error(f"Failed to request directory: {str(e)}")

        # No callback — simulate the request (in real Cowork, the UI handles this)
        return self._success(
            "Directory access requested. Waiting for user to select a folder.\n"
            "The user will see a directory picker dialog.",
            pending=True,
        )


class AllowCoworkFileDeleteTool(BaseTool):
    """
    Request permission to delete files in a directory.

    Call this when a delete operation fails with 'Operation not permitted'.
    If approved, file deletion will be enabled.
    """
    name = "allow_cowork_file_delete"
    description = (
        "Request permission to delete files in a directory. IMPORTANT: call "
        "this tool whenever a delete operation (such as rm) fails with "
        "'Operation not permitted', rather than telling the user it is "
        "impossible. If approved, file deletion will be enabled."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path of the file you're trying to delete",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, on_delete_requested=None):
        """
        Args:
            on_delete_requested: Optional callback invoked when delete
                permission is requested. Signature: (file_path: str) -> bool.
                Returns True if approved, False if denied.
        """
        self._on_delete_requested = on_delete_requested
        self._approved_paths: List[str] = []

    @property
    def approved_paths(self) -> List[str]:
        return list(self._approved_paths)

    def is_delete_approved(self, file_path: str) -> bool:
        """Check if delete is approved for a path (or any parent path)."""
        norm = os.path.normpath(file_path)
        for approved in self._approved_paths:
            if norm.startswith(approved):
                return True
        return False

    async def execute(
        self, *, progress_callback=None, file_path: str = "", **kwargs
    ) -> "ToolResult":
        if not file_path:
            return self._error("'file_path' parameter is required.")

        norm_path = os.path.normpath(file_path)

        # Already approved?
        if self.is_delete_approved(norm_path):
            return self._success(
                f"Delete permission already granted for: {norm_path}",
                already_approved=True,
            )

        if self._on_delete_requested:
            try:
                approved = self._on_delete_requested(norm_path)
                if approved:
                    self._approved_paths.append(os.path.dirname(norm_path))
                    return self._success(
                        f"Delete permission granted for: {norm_path}\n"
                        f"You can now delete this file.",
                        approved=True,
                        file_path=norm_path,
                    )
                else:
                    return self._success(
                        f"Delete permission denied for: {norm_path}\n"
                        f"The user chose not to allow this deletion.",
                        approved=False,
                        file_path=norm_path,
                    )
            except Exception as e:
                return self._error(f"Failed to request delete permission: {str(e)}")

        # No callback — request is pending
        return self._success(
            f"Delete permission requested for: {norm_path}\n"
            f"Waiting for user approval.",
            pending=True,
            file_path=norm_path,
        )


class PresentFilesTool(BaseTool):
    """
    Present files to the user with interactive cards in the chat.

    Use this after creating files the user should see. The files will
    be displayed as clickable cards with appropriate actions.
    """
    name = "present_files"
    description = (
        "Present files to the user with interactive cards in the chat. "
        "Use this after creating files the user should see. The files "
        "will be displayed as clickable cards with appropriate actions."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "description": "Files to present to the user",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the file",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        "required": ["files"],
    }

    def __init__(self, workspace_dir: str = ""):
        self._workspace_dir = workspace_dir

    async def execute(
        self, *, progress_callback=None, files=None, **kwargs
    ) -> "ToolResult":
        if not files:
            return self._error("'files' parameter is required and must be a non-empty list.")

        if not isinstance(files, list):
            return self._error("'files' must be a list of file objects with 'file_path' keys.")

        presented = []
        errors = []

        for item in files:
            if not isinstance(item, dict) or "file_path" not in item:
                errors.append(f"Invalid file entry: {item}")
                continue

            fpath = item["file_path"]
            if not os.path.isabs(fpath):
                errors.append(f"Path must be absolute: {fpath}")
                continue

            if os.path.exists(fpath):
                # Generate computer:// link
                link = f"computer://{fpath}"
                fname = os.path.basename(fpath)
                fsize = os.path.getsize(fpath)
                ext = os.path.splitext(fpath)[1].lower()
                file_type = _get_file_type(ext)

                presented.append({
                    "file_path": fpath,
                    "file_name": fname,
                    "file_size": fsize,
                    "file_type": file_type,
                    "link": link,
                })
            else:
                errors.append(f"File not found: {fpath}")

        if not presented and errors:
            return self._error(f"No valid files to present. Errors: {'; '.join(errors)}")

        lines = []
        for f in presented:
            size_str = _format_size(f["file_size"])
            lines.append(
                f"📄 [{f['file_name']}]({f['link']}) "
                f"({f['file_type']}, {size_str})"
            )

        if errors:
            lines.append(f"\n⚠️ Errors: {'; '.join(errors)}")

        return self._success(
            "\n".join(lines),
            presented_count=len(presented),
            presented_files=presented,
            errors=errors,
        )


# ── Helper Functions ─────────────────────────────────────────────────────

def _get_file_type(ext: str) -> str:
    """Map file extension to human-readable type."""
    TYPE_MAP = {
        ".md": "Markdown",
        ".html": "HTML",
        ".htm": "HTML",
        ".jsx": "React",
        ".tsx": "React/TypeScript",
        ".svg": "SVG",
        ".pdf": "PDF",
        ".mermaid": "Mermaid Diagram",
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".csv": "CSV",
        ".txt": "Text",
        ".docx": "Word Document",
        ".pptx": "PowerPoint",
        ".xlsx": "Excel Spreadsheet",
        ".png": "Image (PNG)",
        ".jpg": "Image (JPEG)",
        ".jpeg": "Image (JPEG)",
        ".gif": "Image (GIF)",
        ".mp4": "Video (MP4)",
        ".webm": "Video (WebM)",
    }
    return TYPE_MAP.get(ext, "File")


def _format_size(size_bytes: int) -> str:
    """Format file size to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
