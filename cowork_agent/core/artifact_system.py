"""
Artifact System — Sprint 32.

Provides rendering classification and computer:// protocol link generation
for files created by the agent. Mirrors real Cowork's artifact rendering
where certain file types get special rendering in the UI.

Renderable artifact types:
  - Markdown (.md)
  - HTML (.html)
  - React (.jsx)
  - Mermaid (.mermaid)
  - SVG (.svg)
  - PDF (.pdf)

Also handles:
  - computer:// protocol link generation
  - File type classification
  - Artifact metadata extraction
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ── Renderable Types ─────────────────────────────────────────────────────

RENDERABLE_EXTENSIONS: Dict[str, str] = {
    ".md": "markdown",
    ".html": "html",
    ".htm": "html",
    ".jsx": "react",
    ".tsx": "react",
    ".mermaid": "mermaid",
    ".svg": "svg",
    ".pdf": "pdf",
}

# React libraries available in artifact sandbox
REACT_AVAILABLE_LIBRARIES = {
    "lucide-react": "0.263.1",
    "recharts": "latest",
    "mathjs": "latest",
    "lodash": "latest",
    "d3": "latest",
    "plotly": "latest",
    "three": "r128",
    "papaparse": "latest",
    "sheetjs": "latest",
    "chart.js": "latest",
    "tone": "latest",
    "mammoth": "latest",
    "tensorflow": "latest",
}


@dataclass
class ArtifactInfo:
    """Metadata about a rendered artifact."""
    file_path: str
    file_name: str
    artifact_type: str  # "markdown", "html", "react", "mermaid", "svg", "pdf", "code", "file"
    extension: str
    size_bytes: int = 0
    computer_link: str = ""
    is_renderable: bool = False
    title: str = ""


class ArtifactSystem:
    """
    Manages artifact classification, rendering, and link generation.

    Determines which files can be rendered in-UI, generates computer://
    protocol links, and tracks artifacts created during a session.
    """

    def __init__(self, workspace_dir: str = "", session_dir: str = ""):
        self._workspace_dir = workspace_dir
        self._session_dir = session_dir
        self._artifacts: Dict[str, ArtifactInfo] = {}  # path -> ArtifactInfo

    @property
    def workspace_dir(self) -> str:
        return self._workspace_dir

    @workspace_dir.setter
    def workspace_dir(self, value: str):
        self._workspace_dir = value

    def classify(self, file_path: str) -> ArtifactInfo:
        """
        Classify a file and determine if it's a renderable artifact.

        Returns ArtifactInfo with type, renderability, and computer:// link.
        """
        ext = os.path.splitext(file_path)[1].lower()
        fname = os.path.basename(file_path)
        artifact_type = RENDERABLE_EXTENSIONS.get(ext, "file")
        is_renderable = ext in RENDERABLE_EXTENSIONS

        # Code files get "code" type
        if not is_renderable and ext in {
            ".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp",
            ".rb", ".php", ".sh", ".bash", ".zsh",
        }:
            artifact_type = "code"

        size = 0
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)

        info = ArtifactInfo(
            file_path=file_path,
            file_name=fname,
            artifact_type=artifact_type,
            extension=ext,
            size_bytes=size,
            computer_link=self.generate_link(file_path),
            is_renderable=is_renderable,
            title=os.path.splitext(fname)[0],
        )

        self._artifacts[file_path] = info
        return info

    def generate_link(self, file_path: str) -> str:
        """
        Generate a computer:// protocol link for a file.

        These links allow the Cowork UI to open files directly.
        The path must be absolute.
        """
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        return f"computer://{file_path}"

    def get_share_link(self, file_path: str) -> str:
        """
        Generate a markdown-formatted link suitable for sharing with the user.

        Format: [View filename](computer:///path/to/file)
        """
        fname = os.path.basename(file_path)
        link = self.generate_link(file_path)
        return f"[View {fname}]({link})"

    def is_renderable(self, file_path: str) -> bool:
        """Check if a file type can be rendered in the Cowork UI."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in RENDERABLE_EXTENSIONS

    def get_artifact_type(self, file_path: str) -> str:
        """Get the artifact rendering type for a file."""
        ext = os.path.splitext(file_path)[1].lower()
        return RENDERABLE_EXTENSIONS.get(ext, "file")

    def suggest_output_path(self, filename: str) -> str:
        """
        Suggest the best output path for a file.

        If workspace_dir is set, files go there (user can see them).
        Otherwise falls back to session_dir.
        """
        if self._workspace_dir:
            return os.path.join(self._workspace_dir, filename)
        elif self._session_dir:
            return os.path.join(self._session_dir, filename)
        return filename

    @property
    def artifacts(self) -> List[ArtifactInfo]:
        """Return all tracked artifacts."""
        return list(self._artifacts.values())

    @property
    def renderable_artifacts(self) -> List[ArtifactInfo]:
        """Return only renderable artifacts."""
        return [a for a in self._artifacts.values() if a.is_renderable]

    def clear(self) -> None:
        """Clear tracked artifacts."""
        self._artifacts.clear()

    def __len__(self) -> int:
        return len(self._artifacts)

    @staticmethod
    def get_renderable_extensions() -> Dict[str, str]:
        """Return the mapping of renderable extensions to types."""
        return dict(RENDERABLE_EXTENSIONS)

    @staticmethod
    def get_react_libraries() -> Dict[str, str]:
        """Return available React libraries for artifact sandbox."""
        return dict(REACT_AVAILABLE_LIBRARIES)
